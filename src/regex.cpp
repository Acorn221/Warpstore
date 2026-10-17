// Regular expressions for REGEXP / REGEXP_LIKE / REGEXP_REPLACE: RE2 (linear time), compiled
// patterns cached per thread. In its own file because nvcc can't parse re2.h.
#include <re2/re2.h>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

#include "common.h"

namespace gdb {

namespace {
const RE2& cachedRegex(const std::string& pat, bool ci) {
  thread_local std::unordered_map<std::string, std::unique_ptr<RE2>> cache;
  const std::string key = (ci ? "i:" : "c:") + pat;
  auto it = cache.find(key);
  if (it != cache.end()) return *it->second;
  RE2::Options o;
  o.set_case_sensitive(!ci);
  o.set_log_errors(false);
  auto re = std::make_unique<RE2>(pat, o);
  if (!re->ok()) throw SqlError("Invalid regular expression: " + pat + " (" + re->error() + ")", 1139);
  if (cache.size() > 256) cache.clear();
  return *cache.emplace(key, std::move(re)).first->second;
}
}  // namespace

bool regexLike(const std::string& s, const std::string& pat, bool ci) { return RE2::PartialMatch(s, cachedRegex(pat, ci)); }

// REGEXP_SUBSTR: the first match (false: none); REGEXP_INSTR: its 1-based position (0: none).
bool regexSubstr(const std::string& s, const std::string& pat, bool ci, std::string& out, int64_t& pos) {
  re2::StringPiece in(s), m;
  if (!cachedRegex(pat, ci).Match(in, 0, s.size(), RE2::UNANCHORED, &m, 1)) return false;
  out.assign(m.data(), m.size());
  pos = (int64_t)(m.data() - s.data()) + 1;
  return true;
}

#ifdef GDB_PCRE2
extern "C" {  // the PCRE2 C API (8-bit), declared here so only the runtime library is needed
typedef struct pcre2_real_code_8 pcre2_code_8;
typedef struct pcre2_real_match_data_8 pcre2_match_data_8;
typedef struct pcre2_real_match_context_8 pcre2_match_context_8;
pcre2_code_8* pcre2_compile_8(const uint8_t*, size_t, uint32_t, int*, size_t*, void*);
int pcre2_jit_compile_8(pcre2_code_8*, uint32_t);
void pcre2_code_free_8(pcre2_code_8*);
pcre2_match_data_8* pcre2_match_data_create_from_pattern_8(const pcre2_code_8*, void*);
void pcre2_match_data_free_8(pcre2_match_data_8*);
pcre2_match_context_8* pcre2_match_context_create_8(void*);
int pcre2_set_match_limit_8(pcre2_match_context_8*, uint32_t);
int pcre2_match_8(const pcre2_code_8*, const uint8_t*, size_t, size_t, uint32_t, pcre2_match_data_8*, pcre2_match_context_8*);
size_t* pcre2_get_ovector_pointer_8(pcre2_match_data_8*);
int pcre2_substitute_8(const pcre2_code_8*, const uint8_t*, size_t, size_t, uint32_t, pcre2_match_data_8*, pcre2_match_context_8*,
                       const uint8_t*, size_t, uint8_t*, size_t*);
}
namespace {
struct Pcre {
  pcre2_code_8* code = nullptr;
  pcre2_match_data_8* md = nullptr;
  pcre2_match_context_8* mc = nullptr;
  ~Pcre() {
    if (md) pcre2_match_data_free_8(md);
    if (code) pcre2_code_free_8(code);
  }
};
// Compiled + JIT-compiled per thread and pattern; null when PCRE2 rejects the pattern (RE2 then).
Pcre* cachedPcre(const std::string& pat) {
  thread_local std::unordered_map<std::string, std::unique_ptr<Pcre>> cache;
  thread_local std::string lastPat;
  thread_local Pcre* last = nullptr;
  if (last && pat == lastPat) return last;
  auto it = cache.find(pat);
  if (it != cache.end()) {
    lastPat = pat, last = it->second.get();
    return last;
  }
  if (it != cache.end()) return it->second.get();
  auto p = std::make_unique<Pcre>();
  int err;
  size_t off;
  p->code = pcre2_compile_8((const uint8_t*)pat.data(), pat.size(), 0x00080000u /* UTF */ | 0x00000010u /* DOLLAR_ENDONLY: $ as in RE2 */, &err, &off, nullptr);
  if (p->code) {
    pcre2_jit_compile_8(p->code, 1 /* JIT_COMPLETE */);
    p->md = pcre2_match_data_create_from_pattern_8(p->code, nullptr);
    p->mc = pcre2_match_context_create_8(nullptr);
    pcre2_set_match_limit_8(p->mc, 100000);  // backtracking guard: RE2 takes over beyond it
  }
  if (cache.size() > 256) cache.clear();
  Pcre* r = p->code ? p.get() : nullptr;
  cache.emplace(pat, std::move(p));
  lastPat = pat, last = r;
  return r;
}
}  // namespace
#endif

std::string regexReplace(const std::string& s, const std::string& pat, const std::string& rep) {
#ifdef GDB_PCRE2
  if (Pcre* p = cachedPcre(pat)) {
    thread_local std::string lastRep, prep;  // \N and $N -> ${N}; a literal $ -> $$
    if (rep != lastRep) {
      lastRep = rep;
      prep.clear();
      for (size_t i = 0; i < rep.size(); ++i) {
        if ((rep[i] == '\\' || rep[i] == '$') && i + 1 < rep.size() && isdigit((unsigned char)rep[i + 1])) {
          size_t j = i + 1;
          std::string n;
          while (j < rep.size() && isdigit((unsigned char)rep[j])) n += rep[j++];
          prep += "${" + n + "}";
          i = j - 1;
        } else if (rep[i] == '\\' && i + 1 < rep.size()) {
          prep += rep[++i] == '$' ? std::string("$$") : std::string(1, rep[i]);
        } else {
          prep += rep[i] == '$' ? std::string("$$") : std::string(1, rep[i]);
        }
      }
    }
    // Anchored with ^ (no multiline): at most one match, so one match call and the groups spliced in
    // directly (pcre2_substitute costs about twice a match).
    if (!pat.empty() && pat[0] == '^' && pat.find('|') == std::string::npos) {  // (an alternative could match anywhere)
      thread_local std::string lastPrep;
      thread_local std::vector<std::pair<std::string, int>> pieces;  // literal text, then group (-1: none)
      if (prep != lastPrep) {
        lastPrep = prep;
        pieces.clear();
        std::string lit;
        for (size_t i = 0; i < prep.size(); ++i) {
          if (prep[i] == '$' && i + 1 < prep.size() && prep[i + 1] == '$') { lit += '$'; ++i; continue; }
          if (prep[i] == '$' && i + 1 < prep.size() && prep[i + 1] == '{') {
            size_t e = prep.find('}', i);
            pieces.push_back({lit, std::atoi(prep.substr(i + 2, e - i - 2).c_str())});
            lit.clear();
            i = e;
            continue;
          }
          lit += prep[i];
        }
        pieces.push_back({lit, -1});
      }
      const int rc = pcre2_match_8(p->code, (const uint8_t*)s.data(), s.size(), 0, 0, p->md, p->mc);
      if (rc == -1 /* NOMATCH */) return s;
      if (rc > 0) {
        const size_t* ov = pcre2_get_ovector_pointer_8(p->md);
        std::string out;
        out.reserve(s.size());
        out.append(s, 0, ov[0]);
        for (auto& [lit, g] : pieces) {
          out += lit;
          if (g >= 0 && g < rc && ov[2 * g] != ~(size_t)0) out.append(s, ov[2 * g], ov[2 * g + 1] - ov[2 * g]);
        }
        out.append(s, ov[1], std::string::npos);
        return out;
      }
      // errors (bad UTF-8, match limit): RE2 below
    }
    thread_local std::string out;
    out.resize(s.size() + 64);
    for (int attempt = 0; attempt < 2; ++attempt) {
      size_t len = out.size();
      int rc = pcre2_substitute_8(p->code, (const uint8_t*)s.data(), s.size(), 0, 0x00000100u /* GLOBAL */ | 0x00001000u /* OVERFLOW_LENGTH */,
                                  p->md, p->mc, (const uint8_t*)prep.data(), prep.size(), (uint8_t*)out.data(), &len);
      if (rc >= 0) return out.substr(0, len);
      if (rc != -48 /* NOMEMORY: len is the size needed */) break;  // other errors (bad UTF-8, match limit): RE2
      out.assign(len + 1, '\0');
    }
  }
#endif
  std::string out = s, rw;
  for (size_t i = 0; i < rep.size(); ++i) {  // $N (MySQL / ICU) and \N both refer to group N
    if (rep[i] == '$' && i + 1 < rep.size() && isdigit((unsigned char)rep[i + 1])) rw += '\\';
    else rw += rep[i];
  }
  RE2::GlobalReplace(&out, cachedRegex(pat, false), rw);
  return out;
}

// ---- CONVERT_TZ: '+05:30' style offsets, UTC / GMT / SYSTEM (the server runs in UTC), and IANA
// names (America/New_York) through the system time zone database.
bool tzOffset(const std::string& zone, int64_t secs, bool local, int64_t& off) {
  std::string z = zone;
  for (auto& ch : z) ch = (char)toupper((unsigned char)ch);
  if (z == "UTC" || z == "GMT" || z == "Z" || z == "SYSTEM" || z == "+00:00") return off = 0, true;
  if (!zone.empty() && (zone[0] == '+' || zone[0] == '-')) {
    int h = 0, m = 0;
    if (sscanf(zone.c_str() + 1, "%d:%d", &h, &m) < 1) return false;
    off = (zone[0] == '-' ? -1 : 1) * (h * 3600 + m * 60);
    return true;
  }
  try {
    const auto* tz = std::chrono::locate_zone(zone);
    if (local) {
      auto info = tz->get_info(std::chrono::local_seconds(std::chrono::seconds(secs)));
      off = info.first.offset.count();
    } else {
      off = tz->get_info(std::chrono::sys_seconds(std::chrono::seconds(secs))).offset.count();
    }
    return true;
  } catch (...) {
    return false;
  }
}

// ---- GEOGRAPHY functions on WKT text (POINT(lon lat), POLYGON((lon lat, ..))), spherical earth.
namespace {
bool wktPoint(const std::string& s, double& lon, double& lat) {
  const size_t p = s.find('(');
  return p != std::string::npos && sscanf(s.c_str() + p + 1, "%lf %lf", &lon, &lat) == 2;
}
std::vector<std::pair<double, double>> wktRing(const std::string& s) {
  std::vector<std::pair<double, double>> pts;
  size_t p = s.find("((");
  if (p == std::string::npos) return pts;
  p += 2;
  while (p < s.size() && s[p] != ')') {
    double x, y;
    int used = 0;
    if (sscanf(s.c_str() + p, " %lf %lf%n", &x, &y, &used) != 2) break;
    pts.push_back({x, y});
    p += used;
    while (p < s.size() && (s[p] == ',' || s[p] == ' ')) ++p;
  }
  return pts;
}
double haversine(double lon1, double lat1, double lon2, double lat2) {
  const double r = 6371008.8, k = M_PI / 180;
  const double dlat = (lat2 - lat1) * k, dlon = (lon2 - lon1) * k;
  const double h = std::sin(dlat / 2) * std::sin(dlat / 2) + std::cos(lat1 * k) * std::cos(lat2 * k) * std::sin(dlon / 2) * std::sin(dlon / 2);
  return 2 * r * std::asin(std::min(1.0, std::sqrt(h)));
}
bool inRing(const std::vector<std::pair<double, double>>& ring, double x, double y) {
  bool in = false;
  for (size_t i = 0, j = ring.size() - 1; i < ring.size(); j = i++)
    if ((ring[i].second > y) != (ring[j].second > y) &&
        x < (ring[j].first - ring[i].first) * (y - ring[i].second) / (ring[j].second - ring[i].second) + ring[i].first)
      in = !in;
  return in;
}
std::string fmtNum(double v) {
  char b[40];
  snprintf(b, sizeof b, "%.8g", v);
  return b;
}
}  // namespace

Value geoFunction(const std::string& f, const std::vector<Value>& a) {
  auto s = [&](size_t i) { return a[i].toString(); };
  double x1, y1, x2, y2;
  if (f == "GEOGRAPHY_POINT" && a.size() == 2) return Value::S("POINT(" + fmtNum(a[0].num()) + " " + fmtNum(a[1].num()) + ")");
  if ((f == "GEOGRAPHY_LONGITUDE" || f == "GEOGRAPHY_LATITUDE") && a.size() == 1) {
    if (!wktPoint(s(0), x1, y1)) return Value::Null();
    return Value::D(f == "GEOGRAPHY_LONGITUDE" ? x1 : y1);
  }
  if ((f == "GEOGRAPHY_DISTANCE" || f == "GEOGRAPHY_WITHIN_DISTANCE") && a.size() >= 2) {
    if (!wktPoint(s(0), x1, y1) || !wktPoint(s(1), x2, y2)) return Value::Null();
    const double d = haversine(x1, y1, x2, y2);
    if (f == "GEOGRAPHY_DISTANCE") return Value::D(d);
    return Value::I(a.size() > 2 && d <= a[2].num());
  }
  if ((f == "GEOGRAPHY_CONTAINS" || f == "GEOGRAPHY_INTERSECTS") && a.size() == 2) {
    std::string poly = s(0), pt = s(1);
    if (f == "GEOGRAPHY_INTERSECTS" && poly.find("POLYGON") == std::string::npos) std::swap(poly, pt);
    auto ring = wktRing(poly);
    if (ring.size() < 3 || !wktPoint(pt, x1, y1)) return Value::Null();
    return Value::I(inRing(ring, x1, y1));
  }
  throw SqlError("FUNCTION gpudb." + f + " does not exist (or wrong arguments)", 1305);
}

}  // namespace gdb
