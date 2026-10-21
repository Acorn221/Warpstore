// MySQL client/server protocol (v10 handshake, text protocol) so any MySQL client, driver or
// BI tool can talk to warpstore, the same way SingleStore is MySQL wire compatible.
#include <arpa/inet.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <set>
#include <regex>
#include <string>
#include <vector>
#include <thread>

#include "engine.h"
#include "monitor.h"
#include "storage.h"
#include "gpuio.h"
#include "jit.h"
#include "gpu.h"

using namespace gdb;

namespace {

std::atomic<uint32_t> g_connId{1};
bool g_verbose = false;
std::string g_dataDir;  // LOAD DATA LOCAL uploads are kept here (the WAL replays them)

void putInt(std::string& b, uint64_t v, int n) {
  for (int i = 0; i < n; ++i) b += (char)((v >> (8 * i)) & 0xff);
}
void putLenenc(std::string& b, uint64_t v) {
  if (v < 251) b += (char)v;
  else if (v < (1 << 16)) { b += (char)0xfc; putInt(b, v, 2); }
  else if (v < (1 << 24)) { b += (char)0xfd; putInt(b, v, 3); }
  else { b += (char)0xfe; putInt(b, v, 8); }
}
void putLenStr(std::string& b, const std::string& s) {
  putLenenc(b, s.size());
  b += s;
}

struct Conn {
  int fd;
  uint8_t seq = 0;
  std::string out;
  SSL* ssl = nullptr;  // after the client's SSL request: everything goes through TLS
  ~Conn() {
    if (ssl) {
      SSL_shutdown(ssl);
      SSL_free(ssl);
    }
  }

  bool readFull(void* p, size_t n) {
    char* c = (char*)p;
    while (n) {
      ssize_t r = ssl ? (ssize_t)SSL_read(ssl, c, (int)std::min<size_t>(n, 1 << 30)) : ::read(fd, c, n);
      if (r <= 0) return false;
      c += r;
      n -= (size_t)r;
    }
    return true;
  }
  bool readPacket(std::string& payload) {
    payload.clear();
    for (;;) {
      uint8_t h[4];
      if (!readFull(h, 4)) return false;
      size_t len = h[0] | (h[1] << 8) | (h[2] << 16);
      seq = h[3] + 1;
      size_t o = payload.size();
      payload.resize(o + len);
      if (len && !readFull(&payload[o], len)) return false;
      if (len < 0xffffff) return true;
    }
  }
  void packet(const std::string& p) {
    size_t off = 0;
    do {
      size_t len = std::min<size_t>(p.size() - off, 0xffffff);
      putInt(out, len, 3);
      out += (char)seq++;
      out.append(p, off, len);
      off += len;
      if (len < 0xffffff) break;
    } while (true);
    if (out.size() > (1 << 20)) flush();
  }
  bool flush() {
    size_t o = 0;
    while (o < out.size()) {
      ssize_t w = ssl ? (ssize_t)SSL_write(ssl, out.data() + o, (int)std::min<size_t>(out.size() - o, 1 << 30))
                      : ::write(fd, out.data() + o, out.size() - o);
      if (w <= 0) { out.clear(); return false; }
      o += (size_t)w;
    }
    out.clear();
    return true;
  }
  uint16_t more = 0;  // SERVER_MORE_RESULTS_EXISTS while answering a multi-statement query
  void ok(uint64_t affected = 0, const std::string& info = "", uint64_t insertId = 0) {
    std::string p;
    p += (char)0x00;
    putLenenc(p, affected);
    putLenenc(p, insertId);
    putInt(p, 0x0002 | more, 2);  // SERVER_STATUS_AUTOCOMMIT
    putInt(p, 0, 2);
    p += info;
    packet(p);
  }
  void eof(bool last = false) {
    std::string p;
    p += (char)0xfe;
    putInt(p, 0, 2);
    putInt(p, 0x0002 | (last ? more : 0), 2);
    packet(p);
  }
  void err(int code, const std::string& msg) {
    const char* state = code == 1064 ? "42000" : code == 1062 ? "23000" : code == 1146 ? "42S02" : code == 1054 ? "42S22" : "HY000";
    std::string p;
    p += (char)0xff;
    putInt(p, code, 2);
    p += '#';
    p += state;
    p += msg;
    packet(p);
  }
  void colDef(const ResultSet& r, size_t i) {
    std::string p;
    {
      putLenStr(p, "def");
      putLenStr(p, "warpstore");
      putLenStr(p, "");
      putLenStr(p, "");
      putLenStr(p, r.names[i]);
      putLenStr(p, r.names[i]);
      p += (char)0x0c;
      uint8_t type;
      uint16_t charset = 63, flags = 0x0080 /*BINARY*/;
      uint32_t len;
      uint8_t decimals = 0;
      const bool json = i < r.jsonCols.size() && r.jsonCols[i];
      if (json) { type = 0xf5; len = 4294967295u; charset = 63; flags = 0x0090; }  // JSON (BLOB | BINARY)
      else switch (r.types[i]) {
        case LType::INT:
        case LType::BIGINT: type = 0x08; len = 20; flags |= 0x8000; break;  // LONGLONG, NUM
        case LType::DOUBLE: type = 0x05; len = 22; decimals = 31; flags |= 0x8000; break;
        case LType::DECIMAL: type = 0xf6; len = 66; decimals = 31; flags |= 0x8000; break;
        case LType::DATE: type = 0x0a; len = 10; break;
        case LType::DATETIME: type = 0x0c; len = 26; break;
        default: type = 0xfd; len = 1024; charset = 33; flags = 0; break;  // VAR_STRING utf8
      }
      putInt(p, charset, 2);
      putInt(p, len, 4);
      p += (char)type;
      putInt(p, flags, 2);
      p += (char)decimals;
      putInt(p, 0, 2);
      packet(p);
    }
  }
  void result(const ResultSet& r, bool binary = false) {
    std::string p;
    putLenenc(p, r.names.size());
    packet(p);
    for (size_t i = 0; i < r.names.size(); ++i) colDef(r, i);
    eof();
    for (auto& row : r.rows) {
      p.clear();
      if (binary) {  // binary protocol row (prepared statements): header, NULL bitmap (offset 2), values
        p += (char)0x00;
        size_t nb = (row.size() + 7 + 2) / 8, base = p.size();
        p.append(nb, '\0');
        for (size_t i = 0; i < row.size(); ++i) {
          const std::string& c = row[i];
          if (c == kNullCell) {
            p[base + (i + 2) / 8] |= (char)(1 << ((i + 2) % 8));
            continue;
          }
          switch (r.types[i]) {
            case LType::INT:
            case LType::BIGINT: putInt(p, (uint64_t)std::strtoll(c.c_str(), nullptr, 10), 8); break;
            case LType::DOUBLE: {
              double d = std::strtod(c.c_str(), nullptr);
              uint64_t u;
              memcpy(&u, &d, 8);
              putInt(p, u, 8);
              break;
            }
            case LType::DATE:
            case LType::DATETIME: {
              int y = 0, mo = 0, d = 0, h = 0, mi = 0, se = 0, us = 0;
              sscanf(c.c_str(), "%d-%d-%d %d:%d:%d.%d", &y, &mo, &d, &h, &mi, &se, &us);
              if (r.types[i] == LType::DATE || (h == 0 && mi == 0 && se == 0 && us == 0)) {
                p += (char)4;
                putInt(p, y, 2);
                p += (char)mo;
                p += (char)d;
              } else {
                p += (char)(us ? 11 : 7);
                putInt(p, y, 2);
                p += (char)mo;
                p += (char)d;
                p += (char)h;
                p += (char)mi;
                p += (char)se;
                if (us) putInt(p, us, 4);
              }
              break;
            }
            default: putLenStr(p, c);
          }
        }
      } else {
        for (auto& c : row) {
          if (c == kNullCell) p += (char)0xfb;
          else putLenStr(p, c);
        }
      }
      packet(p);
    }
    eof(true);
  }
};

// Splits a multi-statement query on top-level ';' (outside quotes and comments).
std::vector<std::string> splitStatements(const std::string& sql) {
  std::vector<std::string> out;
  std::string cur;
  char q = 0;
  int routine = 0, depth = 0;  // inside CREATE PROCEDURE/FUNCTION (2: saw BEGIN), BEGIN/CASE nesting
  for (size_t i = 0; i < sql.size(); ++i) {
    char c = sql[i];
    if (q) {
      cur += c;
      if (c == '\\' && q != '`' && i + 1 < sql.size()) cur += sql[++i];
      else if (c == q) q = 0;
      continue;
    }
    if (c == '\'' || c == '"' || c == '`') { q = c; cur += c; continue; }
    if (c == '-' && i + 1 < sql.size() && sql[i + 1] == '-') { while (i < sql.size() && sql[i] != '\n') ++i; continue; }
    if (c == '#') { while (i < sql.size() && sql[i] != '\n') ++i; continue; }
    if (c == '/' && i + 1 < sql.size() && sql[i + 1] == '*') {
      size_t e = sql.find("*/", i + 2);
      if (i + 2 < sql.size() && sql[i + 2] == '!') {  // /*!40101 ... */ executable comment: keep the body
        size_t b = i + 3;
        while (b < sql.size() && isdigit((unsigned char)sql[b])) ++b;
        cur += sql.substr(b, (e == std::string::npos ? sql.size() : e) - b);
      }
      i = e == std::string::npos ? sql.size() : e + 1;
      continue;
    }
    if (isalpha((unsigned char)c) && (cur.empty() || !(isalnum((unsigned char)cur.back()) || cur.back() == '_'))) {
      size_t j = i;
      while (j < sql.size() && (isalnum((unsigned char)sql[j]) || sql[j] == '_')) ++j;
      std::string w = sql.substr(i, j - i);
      for (auto& ch : w) ch = (char)toupper((unsigned char)ch);
      // Routine bodies: ';' only ends the CREATE PROCEDURE/FUNCTION after its outermost END.
      if (routine == 0 && (w == "PROCEDURE" || w == "FUNCTION")) {
        std::string head = cur;
        for (auto& ch : head) ch = (char)toupper((unsigned char)ch);
        if (head.find("CREATE") != std::string::npos) routine = 1;
      } else if (routine) {
        size_t k = j;
        while (k < sql.size() && isspace((unsigned char)sql[k])) ++k;
        std::string nxt;
        while (k < sql.size() && isalpha((unsigned char)sql[k])) nxt += (char)toupper((unsigned char)sql[k++]);
        if (w == "BEGIN" && nxt != "TRANSACTION" && nxt != "WORK") ++depth, routine = 2;
        else if (w == "RETURN" && routine == 1) routine = 3;  // MySQL single-statement body
        else if (w == "END" && nxt != "IF" && nxt != "LOOP" && nxt != "WHILE" && nxt != "REPEAT" && nxt != "CASE") --depth;
        else if (w == "CASE") ++depth;
      }
      cur += w.size() == j - i ? sql.substr(i, j - i) : w;
      i = j - 1;
      continue;
    }
    if (c == ';' && (routine == 1 || (routine == 2 && depth > 0))) {
      cur += c;
      continue;
    }
    if (c == ';') {
      routine = 0;
      depth = 0;
      out.push_back(cur);
      cur.clear();
      continue;
    }
    cur += c;
  }
  out.push_back(cur);
  std::vector<std::string> nonEmpty;
  for (auto& x : out) {
    size_t a = x.find_first_not_of(" \t\r\n");
    if (a != std::string::npos) nonEmpty.push_back(x.substr(a));
  }
  return nonEmpty;
}

struct Prepared {
  std::string sql;
  int nparams = 0;
  std::vector<uint16_t> types;
  std::vector<std::string> longData;
};

int countParams(const std::string& sql) {
  int n = 0;
  char q = 0;
  for (size_t i = 0; i < sql.size(); ++i) {
    char c = sql[i];
    if (q) {
      if (c == '\\' && q != '`') ++i;
      else if (c == q) q = 0;
    } else if (c == '\'' || c == '"' || c == '`') {
      q = c;
    } else if (c == '?') {
      ++n;
    }
  }
  return n;
}

std::string quoteSql(const std::string& v) {
  std::string r = "'";
  for (char c : v) {
    if (c == '\'') r += "''";
    else if (c == '\\') r += "\\\\";
    else r += c;
  }
  return r + "'";
}

// Substitutes the binary COM_STMT_EXECUTE parameters into the statement text.
std::string bindParams(Prepared& ps, const std::string& pkt) {
  size_t i = 1 + 4 + 1 + 4;  // cmd, stmt id, flags, iteration count
  std::vector<std::string> vals(ps.nparams);
  if (ps.nparams > 0) {
    size_t nb = (ps.nparams + 7) / 8;
    std::string nullmap = pkt.substr(i, nb);
    i += nb;
    bool bound = i < pkt.size() && pkt[i++] == 1;
    if (bound) {
      ps.types.assign(ps.nparams, 0);
      for (int k = 0; k < ps.nparams; ++k) {
        ps.types[k] = (uint8_t)pkt[i] | ((uint8_t)pkt[i + 1] << 8);
        i += 2;
      }
    }
    auto rd = [&](int n) {
      uint64_t v = 0;
      for (int k = 0; k < n && i < pkt.size(); ++k) v |= (uint64_t)(uint8_t)pkt[i++] << (8 * k);
      return v;
    };
    auto lenenc = [&]() -> uint64_t {
      uint8_t f = (uint8_t)pkt[i++];
      if (f < 251) return f;
      if (f == 0xfc) return rd(2);
      if (f == 0xfd) return rd(3);
      return rd(8);
    };
    for (int k = 0; k < ps.nparams; ++k) {
      if ((nullmap[k / 8] >> (k % 8)) & 1) { vals[k] = "NULL"; continue; }
      if (k < (int)ps.longData.size() && !ps.longData[k].empty()) { vals[k] = quoteSql(ps.longData[k]); continue; }
      const uint16_t ty = k < (int)ps.types.size() ? ps.types[k] : 0xfd;
      const bool uns = ty & 0x8000;
      switch (ty & 0xff) {
        case 0x01: vals[k] = std::to_string(uns ? (int64_t)(uint8_t)rd(1) : (int64_t)(int8_t)rd(1)); break;
        case 0x02: case 0x0d: vals[k] = std::to_string(uns ? (int64_t)(uint16_t)rd(2) : (int64_t)(int16_t)rd(2)); break;
        case 0x03: case 0x09: vals[k] = std::to_string(uns ? (int64_t)(uint32_t)rd(4) : (int64_t)(int32_t)rd(4)); break;
        case 0x08: { uint64_t v = rd(8); vals[k] = uns ? std::to_string(v) : std::to_string((int64_t)v); break; }
        case 0x04: { uint32_t u = (uint32_t)rd(4); float f; memcpy(&f, &u, 4); vals[k] = std::to_string(f); break; }
        case 0x05: { uint64_t u = rd(8); double d; memcpy(&d, &u, 8); char b[40]; snprintf(b, sizeof b, "%.17g", d); vals[k] = b; break; }
        case 0x06: vals[k] = "NULL"; break;
        case 0x07: case 0x0a: case 0x0c: {  // TIMESTAMP, DATE, DATETIME
          uint8_t len = (uint8_t)pkt[i++];
          int y = 0, mo = 0, d = 0, h = 0, mi = 0, se = 0, us = 0;
          if (len >= 4) { y = (int)rd(2); mo = (int)rd(1); d = (int)rd(1); }
          if (len >= 7) { h = (int)rd(1); mi = (int)rd(1); se = (int)rd(1); }
          if (len >= 11) us = (int)rd(4);
          char b[48];
          snprintf(b, sizeof b, "'%04d-%02d-%02d %02d:%02d:%02d.%06d'", y, mo, d, h, mi, se, us);
          vals[k] = b;
          break;
        }
        case 0x0b: {  // TIME
          uint8_t len = (uint8_t)pkt[i++];
          i += len;
          vals[k] = "'00:00:00'";
          break;
        }
        default: {  // strings, decimals, blobs, JSON
          uint64_t n = lenenc();
          vals[k] = quoteSql(pkt.substr(i, n));
          i += n;
        }
      }
    }
  }
  std::string out;
  char q = 0;
  int k = 0;
  for (size_t j = 0; j < ps.sql.size(); ++j) {
    char c = ps.sql[j];
    if (q) {
      out += c;
      if (c == '\\' && q != '`' && j + 1 < ps.sql.size()) out += ps.sql[++j];
      else if (c == q) q = 0;
    } else if (c == '\'' || c == '"' || c == '`') {
      q = c;
      out += c;
    } else if (c == '?' && k < ps.nparams) {
      out += vals[k++];
    } else {
      out += c;
    }
  }
  return out;
}

// ---------------------------------------------------------------- HTTP Data API (SingleStore Data API compatible endpoints)
// POST /api/v2/query/rows   {"sql": "...", "args": [..], "database": "db"} -> {"results":[{"rows":[{col: val}]}]}
// POST /api/v2/query/tuples                                              -> {"results":[{"columns":[..],"rows":[[..]]}]}
// POST /api/v2/exec                                                      -> {"lastInsertId":N,"rowsAffected":N}
// GET  /ping
struct Json {
  enum K { NUL, BOOL, NUM, STR, ARR, OBJ } k = NUL;
  bool b = false;
  std::string s;  // number text or string
  std::vector<Json> a;
  std::vector<std::pair<std::string, Json>> o;
  const Json* get(const std::string& key) const {
    for (auto& [n, v] : o)
      if (n == key) return &v;
    return nullptr;
  }
};
bool parseJson(const std::string& t, size_t& i, Json& out) {
  auto ws = [&] { while (i < t.size() && isspace((unsigned char)t[i])) ++i; };
  ws();
  if (i >= t.size()) return false;
  char c = t[i];
  if (c == '{' || c == '[') {
    bool obj = c == '{';
    out.k = obj ? Json::OBJ : Json::ARR;
    ++i;
    ws();
    if (i < t.size() && t[i] == (obj ? '}' : ']')) { ++i; return true; }
    for (;;) {
      std::string key;
      if (obj) {
        Json kj;
        if (!parseJson(t, i, kj) || kj.k != Json::STR) return false;
        key = kj.s;
        ws();
        if (i >= t.size() || t[i] != ':') return false;
        ++i;
      }
      Json v;
      if (!parseJson(t, i, v)) return false;
      if (obj) out.o.push_back({key, std::move(v)});
      else out.a.push_back(std::move(v));
      ws();
      if (i < t.size() && t[i] == ',') { ++i; continue; }
      if (i < t.size() && t[i] == (obj ? '}' : ']')) { ++i; return true; }
      return false;
    }
  }
  if (c == '"') {
    out.k = Json::STR;
    for (++i; i < t.size() && t[i] != '"'; ++i) {
      if (t[i] != '\\') { out.s += t[i]; continue; }
      char e = t[++i];
      if (e == 'n') out.s += '\n';
      else if (e == 't') out.s += '\t';
      else if (e == 'r') out.s += '\r';
      else if (e == 'u') {  // \uXXXX -> UTF-8 (BMP)
        unsigned cp = (unsigned)std::stoul(t.substr(i + 1, 4), nullptr, 16);
        i += 4;
        if (cp < 0x80) out.s += (char)cp;
        else if (cp < 0x800) { out.s += (char)(0xc0 | (cp >> 6)); out.s += (char)(0x80 | (cp & 0x3f)); }
        else { out.s += (char)(0xe0 | (cp >> 12)); out.s += (char)(0x80 | ((cp >> 6) & 0x3f)); out.s += (char)(0x80 | (cp & 0x3f)); }
      } else out.s += e;
    }
    ++i;
    return true;
  }
  if (t.compare(i, 4, "null") == 0) { i += 4; out.k = Json::NUL; return true; }
  if (t.compare(i, 4, "true") == 0) { i += 4; out.k = Json::BOOL; out.b = true; return true; }
  if (t.compare(i, 5, "false") == 0) { i += 5; out.k = Json::BOOL; return true; }
  size_t j = i;
  while (j < t.size() && (isdigit((unsigned char)t[j]) || strchr("+-.eE", t[j]))) ++j;
  if (j == i) return false;
  out.k = Json::NUM;
  out.s = t.substr(i, j - i);
  i = j;
  return true;
}
std::string jsonStr(const std::string& s) {
  std::string r = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { r += '\\'; r += (char)c; }
    else if (c == '\n') r += "\\n";
    else if (c == '\t') r += "\\t";
    else if (c == '\r') r += "\\r";
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); r += b; }
    else r += (char)c;
  }
  return r + "\"";
}
std::string jsonCell(const std::string& c, LType t) {
  if (c == kNullCell) return "null";
  if (t == LType::INT || t == LType::BIGINT || t == LType::DOUBLE) return c.empty() ? "null" : c;
  return jsonStr(c);
}
std::string substituteArgs(const std::string& sql, const Json* args) {
  if (!args || args->k != Json::ARR || args->a.empty()) return sql;
  std::string out;
  size_t k = 0;
  char q = 0;
  for (size_t j = 0; j < sql.size(); ++j) {
    char c = sql[j];
    if (q) {
      out += c;
      if (c == '\\' && q != '`' && j + 1 < sql.size()) out += sql[++j];
      else if (c == q) q = 0;
    } else if (c == '\'' || c == '"' || c == '`') {
      q = c;
      out += c;
    } else if (c == '?' && k < args->a.size()) {
      const Json& v = args->a[k++];
      out += v.k == Json::NUL ? "NULL" : v.k == Json::BOOL ? (v.b ? "1" : "0") : v.k == Json::NUM ? v.s : quoteSql(v.s);
    } else {
      out += c;
    }
  }
  return out;
}

std::string base64Decode(const std::string& in) {
  std::string out;
  int val = 0, bits = -8;
  for (unsigned char c : in) {
    int d = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52
          : c == '+' ? 62 : c == '/' ? 63 : -1;
    if (d < 0) break;
    val = (val << 6) | d;
    bits += 6;
    if (bits >= 0) {
      out += (char)((val >> bits) & 0xff);
      bits -= 8;
    }
  }
  return out;
}

std::string urlDecode(const std::string& s) {
  std::string o;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      o += (char)std::strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
      i += 2;
    } else {
      o += s[i] == '+' ? ' ' : s[i];
    }
  }
  return o;
}

// A SELECT for a cluster aggregator, answered in the columnar binary format (typed values).
std::string columnarQuery(Engine* engine, const Json& req) {
  if (const Json* db = req.get("database")) engine->execute("USE `" + db->s + "`");
  tTypedResult = true;
  ResultSet r;
  try {
    r = engine->execute(req.get("sql")->s);
  } catch (...) {
    tTypedResult = false;
    throw;
  }
  tTypedResult = false;
  return encodeColumnar(r);
}

int gDataPort = 0;

// Cluster data plane (--data-port; default: the HTTP port + 10000; -1 disables it): persistent TCP connections carrying
// length-prefixed binary frames instead of HTTP requests. A frame is u32 path length, path, u64 body
// length, body; the reply is u32 status (200 / 400 / 401 / 404), u64 length, payload. The first
// frame on a connection is "auth" with body "user:password".
static bool readFull(int fd, void* p, size_t n) {
  for (size_t o = 0; o < n;) {
    ssize_t r = ::read(fd, (char*)p + o, n - o);
    if (r <= 0) return false;
    o += (size_t)r;
  }
  return true;
}
static bool writeFull(int fd, const void* p, size_t n) {
  for (size_t o = 0; o < n;) {
    ssize_t w = ::write(fd, (const char*)p + o, n - o);
    if (w <= 0) return false;
    o += (size_t)w;
  }
  return true;
}
void serveData(int fd, Engine* engine) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  bool authed = false;
  for (;;) {
    uint32_t pl;
    uint64_t bl;
    if (!readFull(fd, &pl, 4) || pl > 4096) break;
    std::string path(pl, '\0');
    if (!readFull(fd, path.data(), pl) || !readFull(fd, &bl, 8) || bl > (1ull << 36)) break;
    std::string body(bl, '\0');
    if (!readFull(fd, body.data(), bl)) break;
    uint32_t status = 200;
    std::string out;
    try {
      if (path == "auth") {
        const std::string user = body.substr(0, body.find(':')), pw = body.find(':') == std::string::npos ? "" : body.substr(body.find(':') + 1);
        authed = engine->checkPassword(user, pw);
        if (authed) engine->setUser(user);
        status = authed ? 200 : 401;
      } else if (!authed) {
        status = 401, out = "{\"error\":\"access denied\"}";
      } else if (path.rfind("/insert?", 0) == 0) {
        std::string query = path.substr(path.find('?') + 1);
        auto param = [&](const std::string& k) {
          size_t p = ("&" + query).find("&" + k + "=");
          if (p == std::string::npos) return std::string();
          size_t b = p + k.size() + 1, e = query.find('&', b);
          return urlDecode(query.substr(b, e == std::string::npos ? std::string::npos : e - b));
        };
        out = "{\"rowsAffected\":" + std::to_string(engine->insertBinary(param("database"), param("table"), body)) + "}";
      } else if (path == "/reshuffle") {
        out = engine->reshuffle(body);
      } else if (path == "/query/columnar") {
        Json req;
        size_t ji = 0;
        if (!parseJson(body, ji, req) || req.k != Json::OBJ || !req.get("sql")) throw SqlError("bad request", 1105);
        out = columnarQuery(engine, req);
      } else {
        status = 404, out = "{\"error\":\"not found\"}";
      }
    } catch (const std::exception& e) {
      status = 400, out = "{\"error\":" + jsonStr(e.what()) + "}";
    }
    uint64_t ol = out.size();
    if (!writeFull(fd, &status, 4) || !writeFull(fd, &ol, 8) || !writeFull(fd, out.data(), out.size())) break;
  }
  close(fd);
}

void serveHttp(int fd, Engine* engine) {
  std::string buf;
  char tmp[65536];
  auto reply = [&](int code, const std::string& body, const char* ctype = "application/json") {
    const char* msg = code == 200 ? "OK" : code == 404 ? "Not Found" : code == 400 ? "Bad Request" : code == 401 ? "Unauthorized"
                    : code == 410 ? "Gone" : "Internal Server Error";
    std::string r = "HTTP/1.1 " + std::to_string(code) + " " + msg + "\r\nContent-Type: " + ctype +
                    (code == 401 ? "\r\nWWW-Authenticate: Basic realm=\"warpstore\"" : "") +
                    "\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: keep-alive\r\n\r\n" + body;
    size_t o = 0;
    while (o < r.size()) {
      ssize_t w = ::write(fd, r.data() + o, r.size() - o);
      if (w <= 0) return false;
      o += (size_t)w;
    }
    return true;
  };
  for (;;) {
    size_t he;
    while ((he = buf.find("\r\n\r\n")) == std::string::npos) {
      ssize_t n = ::read(fd, tmp, sizeof tmp);
      if (n <= 0) { close(fd); return; }
      buf.append(tmp, n);
    }
    std::string head = buf.substr(0, he);
    size_t clen = 0;
    {
      std::string lowerHead = head;
      for (auto& ch : lowerHead) ch = (char)tolower((unsigned char)ch);
      size_t p = lowerHead.find("content-length:");
      if (p != std::string::npos) clen = (size_t)std::strtoul(head.c_str() + p + 15, nullptr, 10);
    }
    while (buf.size() < he + 4 + clen) {
      ssize_t n = ::read(fd, tmp, sizeof tmp);
      if (n <= 0) { close(fd); return; }
      buf.append(tmp, n);
    }
    std::string body = buf.substr(he + 4, clen);
    buf.erase(0, he + 4 + clen);
    std::string method = head.substr(0, head.find(' '));
    size_t ps = head.find(' ') + 1;
    std::string path = head.substr(ps, head.find(' ', ps) - ps);
    if (path == "/ping") { if (!reply(200, "pong\n", "text/plain")) break; continue; }
    if (path == "/metrics") {  // Prometheus (no auth: the HTTP port binds to --bind, loopback by default)
      if (!reply(200, engine->metrics(), "text/plain; version=0.0.4")) break;
      continue;
    }
    // Basic auth (SingleStore Data API style). Without credentials: root, if root has no password.
    std::string authUser = "root", authPw;
    {
      std::string lh = head;
      for (auto& ch : lh) ch = (char)tolower((unsigned char)ch);
      size_t a = lh.find("\r\nauthorization: basic ");
      if (a != std::string::npos) {
        a += 23;  // strlen("\r\nauthorization: basic ")
        std::string up = base64Decode(head.substr(a, head.find("\r\n", a) == std::string::npos ? std::string::npos : head.find("\r\n", a) - a));
        authUser = up.substr(0, up.find(':'));
        authPw = up.find(':') == std::string::npos ? "" : up.substr(up.find(':') + 1);
      }
    }
    if (!engine->checkPassword(authUser, authPw)) {
      if (!reply(401, "{\"error\":\"access denied\"}")) break;
      continue;
    }
    engine->setUser(authUser);
    if (path.rfind("/replication/", 0) == 0) {  // replica bootstrap / WAL tailing (see repl.cpp)
      if (!engine->isSuper(authUser)) { if (!reply(401, "{\"error\":\"replication needs ALL ON *.*\"}")) break; continue; }
      std::string route = path.substr(0, path.find('?')), query = path.find('?') == std::string::npos ? "" : path.substr(path.find('?') + 1);
      auto param = [&](const std::string& k) {
        size_t p = ("&" + query).find("&" + k + "=");
        if (p == std::string::npos) return std::string();
        std::string v = query.substr(p + k.size() + 1, query.find('&', p + k.size() + 1) == std::string::npos ? std::string::npos : query.find('&', p + k.size() + 1) - p - k.size() - 1);
        return urlDecode(v);
      };
      try {
        if (route == "/replication/snapshot") {
          if (!reply(200, engine->replSnapshot())) break;
        } else if (route == "/replication/wal") {
          int status = 200;
          std::string out = engine->replSince(std::strtoull(param("from").c_str(), nullptr, 10), 1000, status, param("id"), param("epoch"));
          if (!reply(status, out, "application/octet-stream")) break;
        } else if (route == "/replication/file") {
          std::string fp = param("path");
          if (!engine->replFileAllowed(fp)) { if (!reply(404, "{\"error\":\"unknown file\"}")) break; continue; }
          FILE* f = fopen(fp.c_str(), "rb");
          if (!f) { if (!reply(404, "{\"error\":\"file is gone\"}")) break; continue; }
          std::string data;
          char b[1 << 16];
          for (size_t n; (n = fread(b, 1, sizeof b, f)) > 0;) data.append(b, n);
          fclose(f);
          if (!reply(200, data, "application/octet-stream")) break;
        } else if (!reply(404, "{\"error\":\"not found\"}")) {
          break;
        }
      } catch (const std::exception& e) {
        if (!reply(500, "{\"error\":" + jsonStr(e.what()) + "}")) break;
      }
      continue;
    }
    if (path.rfind("/api/v2/load?", 0) == 0 && method == "POST") {  // bulk load (cluster aggregator -> leaf)
      std::string query = path.substr(path.find('?') + 1);
      auto param = [&](const std::string& k) {
        size_t p = ("&" + query).find("&" + k + "=");
        if (p == std::string::npos) return std::string();
        size_t b = p + k.size() + 1, e = query.find('&', b);
        return urlDecode(query.substr(b, e == std::string::npos ? std::string::npos : e - b));
      };
      try {
        const std::string dir = g_dataDir.empty() ? std::string("/tmp") : g_dataDir + "/local-loads";
        if (!g_dataDir.empty()) mkdir(dir.c_str(), 0755);
        std::string tmplS = dir + "/warpstore-leafload-XXXXXX";
        std::vector<char> tmpl(tmplS.begin(), tmplS.end());
        tmpl.push_back('\0');
        int tfd = mkstemp(tmpl.data());
        if (tfd < 0) throw SqlError("cannot stage a load", 1105);
        if (::write(tfd, body.data(), body.size()) != (ssize_t)body.size()) { ::close(tfd); throw SqlError("cannot stage a load", 1105); }
        ::close(tfd);
        std::string delim = param("delim");
        if (delim.empty()) delim = "\\t";
        std::string cols = param("cols");
        if (!param("database").empty()) engine->execute("USE `" + param("database") + "`");
        ResultSet r = engine->execute("LOAD DATA INFILE '" + std::string(tmpl.data()) + "' INTO TABLE " + param("table") +
                                      " FIELDS TERMINATED BY '" + delim + "'" + (cols.empty() ? "" : " (" + cols + ")"));
        if (g_dataDir.empty()) unlink(tmpl.data());
        if (!reply(200, "{\"rowsAffected\":" + std::to_string(r.affected) + "}")) break;
      } catch (const std::exception& e) {
        if (!reply(400, "{\"error\":" + jsonStr(e.what()) + "}")) break;
      }
      continue;
    }
    if (path == "/api/v2/dataport") {  // where the cluster data plane listens (0: none)
      if (!reply(200, std::to_string(gDataPort), "text/plain")) break;
      continue;
    }
    if (path.rfind("/api/v2/insert?", 0) == 0 && method == "POST") {  // shuffle target (binary rows)
      std::string query = path.substr(path.find('?') + 1);
      auto param = [&](const std::string& k) {
        size_t p = ("&" + query).find("&" + k + "=");
        if (p == std::string::npos) return std::string();
        size_t b = p + k.size() + 1, e = query.find('&', b);
        return urlDecode(query.substr(b, e == std::string::npos ? std::string::npos : e - b));
      };
      try {
        int64_t n = engine->insertBinary(param("database"), param("table"), body);
        if (!reply(200, "{\"rowsAffected\":" + std::to_string(n) + "}")) break;
      } catch (const std::exception& e) {
        if (!reply(400, "{\"error\":" + jsonStr(e.what()) + "}")) break;
      }
      continue;
    }
    if (path == "/api/v2/reshuffle" && method == "POST") {
      try {
        if (!reply(200, engine->reshuffle(body))) break;
      } catch (const std::exception& e) {
        if (!reply(400, "{\"error\":" + jsonStr(e.what()) + "}")) break;
      }
      continue;
    }
    if (path == "/api/v2/query/columnar" && method == "POST") {
      // Columnar result format for cluster aggregators: "GDB2", u32 ncols, per column u8 type + u16
      // name length + name, u64 nrows, then per column: nrows NULL flags, then the values: int64 for
      // INT / BIGINT / DATE / DATETIME, float64 for DOUBLE / DECIMAL, else u32 length + bytes each.
      Json req;
      size_t ji = 0;
      if (!parseJson(body, ji, req) || req.k != Json::OBJ || !req.get("sql")) { if (!reply(400, "{\"error\":\"bad request\"}")) break; continue; }
      try {
        const std::string out = columnarQuery(engine, req);
        if (!reply(200, out, "application/octet-stream")) break;
      } catch (const std::exception& e) {
        if (!reply(400, "{\"error\":" + jsonStr(e.what()) + "}")) break;
      }
      continue;
    }
    if (path == "/api/v2/query/binary" && method == "POST") {
      // Compact result format for cluster aggregators: "GDB1", u32 ncols, per column u8 type +
      // u16 name length + name, u64 nrows, then per cell u32 length (0xFFFFFFFF = NULL) + bytes.
      Json req;
      size_t ji = 0;
      if (!parseJson(body, ji, req) || req.k != Json::OBJ || !req.get("sql")) { if (!reply(400, "{\"error\":\"bad request\"}")) break; continue; }
      try {
        if (const Json* db = req.get("database")) engine->execute("USE `" + db->s + "`");
        ResultSet r = engine->execute(req.get("sql")->s);
        std::string out = "GDB1";
        auto put32 = [&](uint32_t v) { out.append((const char*)&v, 4); };
        put32((uint32_t)r.names.size());
        for (size_t c = 0; c < r.names.size(); ++c) {
          out += (char)(c < r.types.size() ? (uint8_t)r.types[c] : (uint8_t)LType::VARCHAR);
          uint16_t n = (uint16_t)r.names[c].size();
          out.append((const char*)&n, 2);
          out += r.names[c];
        }
        uint64_t nr = r.rows.size();
        out.append((const char*)&nr, 8);
        for (auto& row : r.rows)
          for (auto& cell : row) {
            if (cell == kNullCell) { put32(0xFFFFFFFFu); continue; }
            put32((uint32_t)cell.size());
            out += cell;
          }
        if (!reply(200, out, "application/octet-stream")) break;
      } catch (const std::exception& e) {
        if (!reply(400, "{\"error\":" + jsonStr(e.what()) + "}")) break;
      }
      continue;
    }
    const bool rows = path == "/api/v2/query/rows", tuples = path == "/api/v2/query/tuples", exec = path == "/api/v2/exec";
    if (method != "POST" || !(rows || tuples || exec)) { if (!reply(404, "{\"error\":\"not found\"}")) break; continue; }
    Json req;
    size_t i = 0;
    if (!parseJson(body, i, req) || req.k != Json::OBJ || !req.get("sql")) {
      if (!reply(400, "{\"error\":\"expected a JSON object with \\\"sql\\\"\"}")) break;
      continue;
    }
    try {
      if (const Json* db = req.get("database")) engine->execute("USE `" + db->s + "`");
      std::string sql = substituteArgs(req.get("sql")->s, req.get("args"));
      ResultSet r = engine->execute(sql);
      std::string out;
      if (exec || !r.isResult) {
        out = "{\"lastInsertId\":" + std::to_string(r.lastInsertId) + ",\"rowsAffected\":" + std::to_string(r.affected) + "}";
      } else if (rows) {
        out = "{\"results\":[{\"rows\":[";
        for (size_t k = 0; k < r.rows.size(); ++k) {
          out += k ? ",{" : "{";
          for (size_t c = 0; c < r.names.size(); ++c) out += (c ? "," : "") + jsonStr(r.names[c]) + ":" + jsonCell(r.rows[k][c], r.types[c]);
          out += "}";
        }
        out += "]}]}";
      } else {
        out = "{\"results\":[{\"columns\":[";
        for (size_t c = 0; c < r.names.size(); ++c)
          out += std::string(c ? "," : "") + "{\"name\":" + jsonStr(r.names[c]) + ",\"dataType\":" + jsonStr(ltypeName(r.types[c])) + "}";
        out += "],\"rows\":[";
        for (size_t k = 0; k < r.rows.size(); ++k) {
          out += k ? ",[" : "[";
          for (size_t c = 0; c < r.names.size(); ++c) out += (c ? "," : "") + jsonCell(r.rows[k][c], r.types[c]);
          out += "]";
        }
        out += "]}]}";
      }
      if (!reply(200, out)) break;
    } catch (const std::exception& e) {
      if (!reply(400, "{\"error\":" + jsonStr(e.what()) + "}")) break;
    }
  }
  close(fd);
}

// ---- Security: TLS, connection limit, failed-login throttling.
static SSL_CTX* g_tls = nullptr;
static bool g_requireTls = false;
static int g_maxConns = 1000;
static std::atomic<int> g_conns{0};
static std::mutex g_failMu;
static std::map<std::string, std::pair<int, std::chrono::steady_clock::time_point>> g_fails;  // ip -> (failures, window start)

static bool isLoopback(const std::string& ip) { return ip.rfind("127.", 0) == 0 || ip == "::1" || ip == "localhost"; }

// A self-signed certificate (as MySQL 8 creates on first start), kept in the data directory.
static bool tlsInit(std::string cert, std::string key, const std::string& dataDir) {
  SSL_CTX* ctx = SSL_CTX_new(TLS_server_method());
  if (!ctx) return false;
  SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
  if (cert.empty() && !dataDir.empty()) {
    cert = dataDir + "/server-cert.pem";
    key = dataDir + "/server-key.pem";
  }
  if (!cert.empty() && access(cert.c_str(), R_OK) != 0) {  // generate it
    EVP_PKEY* pk = EVP_RSA_gen(2048);
    X509* x = X509_new();
    uint64_t serial = 0;
    RAND_bytes((unsigned char*)&serial, sizeof serial);
    ASN1_INTEGER_set_uint64(X509_get_serialNumber(x), serial >> 1);
    X509_gmtime_adj(X509_getm_notBefore(x), 0);
    X509_gmtime_adj(X509_getm_notAfter(x), 10L * 365 * 24 * 3600);
    X509_set_pubkey(x, pk);
    X509_NAME* nm = X509_get_subject_name(x);
    X509_NAME_add_entry_by_txt(nm, "CN", MBSTRING_ASC, (const unsigned char*)"warpstore_auto_generated_cert", -1, -1, 0);
    X509_set_issuer_name(x, nm);
    X509_sign(x, pk, EVP_sha256());
    FILE* kf = fopen(key.c_str(), "wb");
    if (kf) { fchmod(fileno(kf), 0600); PEM_write_PrivateKey(kf, pk, nullptr, nullptr, 0, nullptr, nullptr); fclose(kf); }
    FILE* cf = fopen(cert.c_str(), "wb");
    if (cf) { PEM_write_X509(cf, x); fclose(cf); }
    X509_free(x);
    EVP_PKEY_free(pk);
    fprintf(stderr, "[warpstore] generated a self-signed TLS certificate: %s\n", cert.c_str());
  }
  if (cert.empty() || SSL_CTX_use_certificate_chain_file(ctx, cert.c_str()) != 1 ||
      SSL_CTX_use_PrivateKey_file(ctx, key.c_str(), SSL_FILETYPE_PEM) != 1) {
    SSL_CTX_free(ctx);
    return false;
  }
  g_tls = ctx;
  return true;
}

void serve(int fd, Engine* engine) {
  std::string peer = "?";
  {
    sockaddr_storage sa{};
    socklen_t sl = sizeof sa;
    char buf[INET6_ADDRSTRLEN] = "";
    if (getpeername(fd, (sockaddr*)&sa, &sl) == 0) {
      if (sa.ss_family == AF_INET) inet_ntop(AF_INET, &((sockaddr_in*)&sa)->sin_addr, buf, sizeof buf);
      else if (sa.ss_family == AF_INET6) inet_ntop(AF_INET6, &((sockaddr_in6*)&sa)->sin6_addr, buf, sizeof buf);
      peer = buf;
    }
  }
  struct ConnCount {
    ConnCount() { ++g_conns; }
    ~ConnCount() { --g_conns; }
  } connCount;
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  Conn c{fd};
  std::map<uint32_t, Prepared> stmts;
  uint32_t nextStmt = 1;
  // ---- Handshake v10 (mysql_native_password advertised; caching_sha2_password via auth switch).
  std::string h;
  h += (char)0x0a;
  h += "8.0.32-warpstore";
  h += '\0';
  const uint32_t connId = g_connId++;
  putInt(h, connId, 4);
  std::string scramble(20, '\0');  // per-connection random challenge (mysql_native_password)
  {  // unpredictable challenge (printable, no NULs)
    unsigned char rb[20];
    if (RAND_bytes(rb, sizeof rb) != 1)
      for (auto& b : rb) b = (unsigned char)rand();
    for (int i = 0; i < 20; ++i) scramble[i] = (char)(33 + rb[i] % 94);
  }
  h += scramble.substr(0, 8);  // auth-plugin-data part 1 (8 bytes)
  h += '\0';
  uint32_t caps = 0x1 | 0x2 | 0x4 | 0x8 | 0x80 | 0x200 | 0x2000 | 0x8000 | 0x10000 | 0x20000 | 0x40000 | 0x80000 | 0x200000;
  if (g_tls) caps |= 0x800;  // CLIENT_SSL
  putInt(h, caps & 0xffff, 2);
  h += (char)33;
  putInt(h, 0x0002, 2);
  putInt(h, caps >> 16, 2);
  h += (char)21;
  h += std::string(10, '\0');
  h += scramble.substr(8);  // part 2 (12 bytes) + NUL
  h += '\0';
  h += "mysql_native_password";
  h += '\0';
  c.seq = 0;
  c.packet(h);
  c.flush();
  std::string pkt;
  if (!c.readPacket(pkt)) { close(fd); return; }
  // SSL request (the first 32 bytes of a handshake response with CLIENT_SSL): TLS, then the real response.
  if (g_tls && pkt.size() == 32 && (((uint8_t)pkt[1] << 8) & 0x800)) {
    c.ssl = SSL_new(g_tls);
    SSL_set_fd(c.ssl, fd);
    if (SSL_accept(c.ssl) != 1 || !c.readPacket(pkt)) { close(fd); return; }
  }
  auto refuse = [&](int code, const std::string& msg) {
    c.err(code, msg);
    c.flush();
    close(fd);
  };
  if (g_conns > g_maxConns) { refuse(1040, "Too many connections"); return; }
  if (g_requireTls && !c.ssl && !isLoopback(peer)) {
    refuse(3159, "Connections using insecure transport are prohibited while --require_secure_transport=ON.");
    return;
  }
  {
    std::lock_guard<std::mutex> g(g_failMu);
    auto it = g_fails.find(peer);
    if (it != g_fails.end() && it->second.first >= 10 && std::chrono::steady_clock::now() - it->second.second < std::chrono::seconds(60)) {
      refuse(1129, "Host '" + peer + "' is blocked because of many failed logins; try again in a minute");
      return;
    }
  }
  // Handshake response: caps(4) max_packet(4) charset(1) filler(23) user\0 auth db\0 (CLIENT_CONNECT_WITH_DB)
  if (pkt.size() > 32) {
    uint32_t ccaps = (uint8_t)pkt[0] | ((uint8_t)pkt[1] << 8) | ((uint8_t)pkt[2] << 16) | ((uint32_t)(uint8_t)pkt[3] << 24);
    size_t i = 32;
    std::string user, token;
    while (i < pkt.size() && pkt[i]) user += pkt[i++];
    ++i;
    if (i < pkt.size()) {
      if (ccaps & 0x200000) {  // CLIENT_PLUGIN_AUTH_LENENC_CLIENT_DATA
        size_t len = (uint8_t)pkt[i++];
        token = pkt.substr(i, len);
        i += len;
      } else if (ccaps & 0x8000) {  // CLIENT_SECURE_CONNECTION
        size_t len = (uint8_t)pkt[i++];
        token = pkt.substr(i, len);
        i += len;
      } else {
        while (i < pkt.size() && pkt[i]) token += pkt[i++];
        ++i;
      }
    }
    // Plugin the client answered with (after the database name): caching_sha2_password (MySQL 8+ / 9
    // clients) is verified with the stored SHA256 hash, else the client is switched to native.
    std::string plugin;
    {
      size_t j = i;
      if (ccaps & 0x8) {
        while (j < pkt.size() && pkt[j]) ++j;
        ++j;
      }
      if ((ccaps & 0x80000) && j < pkt.size())
        while (j < pkt.size() && pkt[j]) plugin += pkt[j++];
    }
    bool authOk = false;
    if (plugin == "caching_sha2_password") {
      const int r = engine->checkAuthSha2(user, scramble, token);
      if (r == 1) {
        authOk = true;
        c.packet(std::string("\x01\x03", 2));  // AuthMoreData: fast_auth_success (then OK)
      } else if (r < 0) {  // no SHA256 hash for this user: AuthSwitchRequest to mysql_native_password
        std::string sw;
        sw += (char)0xfe;
        sw += "mysql_native_password";
        sw += '\0';
        sw += scramble;
        sw += '\0';
        c.packet(sw);
        c.flush();
        std::string resp;
        if (!c.readPacket(resp)) { close(fd); return; }
        authOk = engine->checkAuth(user, scramble, resp);
      }
    } else if (plugin.empty() || plugin == "mysql_native_password") {
      authOk = engine->checkAuth(user, scramble, token);
    } else {  // another plugin: switch to native
      std::string sw;
      sw += (char)0xfe;
      sw += "mysql_native_password";
      sw += '\0';
      sw += scramble;
      sw += '\0';
      c.packet(sw);
      c.flush();
      std::string resp;
      if (!c.readPacket(resp)) { close(fd); return; }
      authOk = engine->checkAuth(user, scramble, resp);
    }
    if (!authOk) {
      {
        std::lock_guard<std::mutex> g(g_failMu);
        auto& f = g_fails[peer];
        const auto now = std::chrono::steady_clock::now();
        if (f.first == 0 || now - f.second > std::chrono::seconds(60)) f = {0, now};
        ++f.first;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      c.err(1045, "Access denied for user '" + user + "'@'localhost' (using password: " + (token.empty() ? "NO" : "YES") + ")");
      c.flush();
      close(fd);
      return;
    }
    {
      std::lock_guard<std::mutex> g(g_failMu);
      g_fails.erase(peer);
    }
    engine->setUser(user);
    if ((ccaps & 0x8) && i < pkt.size()) {
      std::string db;
      while (i < pkt.size() && pkt[i]) db += pkt[i++];
      if (!db.empty()) {
        try {
          engine->execute("USE `" + db + "`");
        } catch (SqlError& e) {
          c.err(e.code, e.what());
          c.flush();
          close(fd);
          return;
        }
      }
    }
  }
  c.ok();
  c.flush();
  auto session = monitor::open(connId, engine->sessionUserName(), peer.empty() ? "localhost" : peer, fd);
  tSession = session.get();
  struct SessionEnd {
    uint32_t id;
    ~SessionEnd() {
      monitor::close(id);
      tSession = nullptr;
    }
  } sessionEnd{connId};

  for (;;) {
    if (session->killConn || !c.readPacket(pkt) || pkt.empty()) break;
    uint8_t cmd = (uint8_t)pkt[0];
    c.seq = 1;
    if (cmd == 0x01) break;  // COM_QUIT
    if (cmd == 0x0e) { c.ok(); c.flush(); continue; }  // PING
    if (cmd == 0x02) {  // INIT_DB
      try {
        engine->execute("USE `" + pkt.substr(1) + "`");
        c.ok();
      } catch (SqlError& e) {
        c.err(e.code, e.what());
      }
      c.flush();
      continue;
    }
    if (cmd == 0x04) { c.eof(); c.flush(); continue; }               // FIELD_LIST
    if (cmd == 0x16) {  // COM_STMT_PREPARE
      Prepared ps;
      ps.sql = pkt.substr(1);
      ps.nparams = countParams(ps.sql);
      uint32_t id = nextStmt++;
      // The result columns, as MySQL reports them at PREPARE (some drivers name result fields from
      // these): the statement is planned with NULL parameters in describe-only mode.
      ResultSet desc;
      {
        static const std::set<std::string> reads = {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"};
        const std::string fw = firstWord(ps.sql);
        if (reads.count(fw) || (fw.empty() && ps.sql.find('(') != std::string::npos)) {
          std::string nulls;
          char q = 0;
          for (size_t j = 0; j < ps.sql.size(); ++j) {
            char ch = ps.sql[j];
            if (q) { nulls += ch; if (ch == '\\' && q != '`' && j + 1 < ps.sql.size()) nulls += ps.sql[++j]; else if (ch == q) q = 0; }
            else if (ch == '\'' || ch == '"' || ch == '`') { q = ch; nulls += ch; }
            else if (ch == '?') {  // NULL, except where an integer is required (LIMIT / OFFSET): 0
              size_t k = nulls.find_last_not_of(" \t\n,0123456789");
              std::string w;
              while (k != std::string::npos && k < nulls.size() && isalpha((unsigned char)nulls[k])) w.insert(w.begin(), (char)toupper((unsigned char)nulls[k])), --k;
              nulls += w == "LIMIT" || w == "OFFSET" ? "0" : "NULL";
            }
            else nulls += ch;
          }
          tDescribeOnly = true;
          try { desc = engine->execute(nulls); } catch (...) { desc = ResultSet{}; }
          tDescribeOnly = false;
          if (!desc.isResult) desc = ResultSet{};
        }
      }
      std::string p;
      p += (char)0x00;
      putInt(p, id, 4);
      putInt(p, desc.names.size(), 2);
      putInt(p, ps.nparams, 2);
      p += (char)0x00;
      putInt(p, 0, 2);
      c.packet(p);
      if (ps.nparams > 0) {
        for (int k = 0; k < ps.nparams; ++k) {
          std::string d;
          putLenStr(d, "def");
          putLenStr(d, "");
          putLenStr(d, "");
          putLenStr(d, "");
          putLenStr(d, "?");
          putLenStr(d, "");
          d += (char)0x0c;
          putInt(d, 63, 2);
          putInt(d, 0, 4);
          d += (char)0xfd;
          putInt(d, 0, 2);
          d += (char)0;
          putInt(d, 0, 2);
          c.packet(d);
        }
        c.eof();
      }
      if (!desc.names.empty()) {
        for (size_t k = 0; k < desc.names.size(); ++k) c.colDef(desc, k);
        c.eof();
      }
      stmts[id] = ps;
      c.flush();
      continue;
    }
    if (cmd == 0x18) {  // COM_STMT_SEND_LONG_DATA: no response
      uint32_t id = (uint8_t)pkt[1] | ((uint8_t)pkt[2] << 8) | ((uint8_t)pkt[3] << 16) | ((uint32_t)(uint8_t)pkt[4] << 24);
      int k = (uint8_t)pkt[5] | ((uint8_t)pkt[6] << 8);
      auto it = stmts.find(id);
      if (it != stmts.end()) {
        if ((int)it->second.longData.size() <= k) it->second.longData.resize(k + 1);
        it->second.longData[k] += pkt.substr(7);
      }
      continue;
    }
    if (cmd == 0x19) {  // COM_STMT_CLOSE: no response
      uint32_t id = (uint8_t)pkt[1] | ((uint8_t)pkt[2] << 8) | ((uint8_t)pkt[3] << 16) | ((uint32_t)(uint8_t)pkt[4] << 24);
      stmts.erase(id);
      continue;
    }
    if (cmd == 0x1a) {  // COM_STMT_RESET
      uint32_t id = (uint8_t)pkt[1] | ((uint8_t)pkt[2] << 8) | ((uint8_t)pkt[3] << 16) | ((uint32_t)(uint8_t)pkt[4] << 24);
      if (stmts.count(id)) stmts[id].longData.clear();
      c.ok();
      c.flush();
      continue;
    }
    if (cmd != 0x03 && cmd != 0x17) {
      c.err(1047, "Unsupported command");
      c.flush();
      continue;
    }
    std::vector<std::string> sqls;
    bool binary = false;
    if (cmd == 0x17) {  // COM_STMT_EXECUTE: substitute the parameters, answer in the binary protocol
      uint32_t id = (uint8_t)pkt[1] | ((uint8_t)pkt[2] << 8) | ((uint8_t)pkt[3] << 16) | ((uint32_t)(uint8_t)pkt[4] << 24);
      auto it = stmts.find(id);
      if (it == stmts.end()) {
        c.err(1243, "Unknown prepared statement handler");
        c.flush();
        continue;
      }
      sqls = {bindParams(it->second, pkt)};
      it->second.longData.clear();
      binary = true;
    } else {
      sqls = splitStatements(pkt.substr(1));
      if (sqls.empty()) sqls = {""};
    }
    std::string sql = sqls.back();
    auto t0 = std::chrono::steady_clock::now();
    for (size_t qi = 0; qi < sqls.size(); ++qi) {
      c.more = qi + 1 < sqls.size() ? 0x0008 : 0;
      bool failed = false;
      try {
        // LOAD DATA LOCAL INFILE: ask the client for the file, stage it, load it from there.
        static const std::regex localRe(R"(^\s*LOAD\s+DATA\s+LOCAL\s+INFILE\s+'((?:[^']|'')*)')", std::regex::icase);
        std::smatch mt;
        std::string tmpPath;
        if (firstWord(sqls[qi]) == "LOAD" && std::regex_search(sqls[qi], mt, localRe)) {
          std::string req;
          req += (char)0xfb;
          req += mt[1].str();
          c.packet(req);
          c.flush();
          std::string dir = g_dataDir.empty() ? "/tmp" : g_dataDir + "/local-loads";
          if (!g_dataDir.empty()) mkdir(dir.c_str(), 0755);
          std::string tmplS = dir + "/warpstore-local-XXXXXX";
          std::vector<char> tmpl(tmplS.begin(), tmplS.end());
          tmpl.push_back('\0');
          int tfd = mkstemp(tmpl.data());
          if (tfd < 0) throw SqlError("cannot stage LOCAL INFILE data", 1105);
          tmpPath = tmpl.data();
          std::string chunk;
          for (;;) {
            if (!c.readPacket(chunk)) throw SqlError("client closed during LOCAL INFILE", 1105);
            if (chunk.empty()) break;
            if (::write(tfd, chunk.data(), chunk.size()) != (ssize_t)chunk.size()) break;
          }
          ::close(tfd);
          sqls[qi] = "LOAD DATA INFILE '" + tmpPath + "'" + sqls[qi].substr(mt.position(0) + mt.length(0));
        }
        struct Unlink {
          std::string p;
          ~Unlink() { if (!p.empty()) unlink(p.c_str()); }
        } unl{g_dataDir.empty() ? tmpPath : std::string()};
        struct LocalFlag {
          explicit LocalFlag(bool on) { tLocalInfile = on; }
          ~LocalFlag() { tLocalInfile = false; }
        } localFlag{!tmpPath.empty()};
        const std::string stmtDb = engine->currentDbName();
        monitor::begin(session.get(), sqls[qi], stmtDb);
        const auto ts = std::chrono::steady_clock::now();
        struct StmtEnd {  // accounting, also when the statement throws
          Session* s;
          const std::string& sql;
          const std::string& db;
          std::chrono::steady_clock::time_point t0;
          int64_t rows = 0;
          bool ok = false;
          std::string plan;
          ~StmtEnd() {
            monitor::end(s, sql, db, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), rows, !ok, plan);
          }
        } stmtEnd{session.get(), sqls[qi], stmtDb, ts};
        ResultSet r = engine->execute(sqls[qi]);
        stmtEnd.ok = true;
        stmtEnd.plan = r.info;
        stmtEnd.rows = r.isResult ? (int64_t)(r.rows.size() + r.vals.size() + (r.numCol ? r.numCol->size() : 0)) : r.affected;
        if (r.isResult) {
          const uint16_t outer = c.more;  // several result sets (CALL): all but the last say more follow
          if (!r.extra.empty()) c.more = 0x0008;
          c.result(r, binary);
          for (size_t i = 0; i < r.extra.size(); ++i) {
            c.more = i + 1 < r.extra.size() ? 0x0008 : outer;
            c.result(r.extra[i], binary);
          }
          c.more = outer;
        } else {
          c.ok(r.affected, r.info, r.lastInsertId);
        }
      } catch (SqlError& e) {
        c.err(e.code, e.what());
        failed = true;
      } catch (std::exception& e) {
        c.err(1105, e.what());
        failed = true;
      }
      if (failed) break;  // MySQL stops a multi-statement batch at the first error
    }
    c.more = 0;
    if (g_verbose) {
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      static const int maxLen = envSetting("LOG_FULL") ? 1 << 20 : 200;
      fprintf(stderr, "[%7.2f ms] %.*s\n", ms, maxLen, sql.c_str());
    }
    if (!c.flush()) break;
  }
  engine->connectionClosed();  // rolls back an open transaction, drops TEMPORARY tables
  close(fd);
}

}  // namespace

namespace gdb { extern int g_redundancy; }
using gdb::g_redundancy;

int main(int argc, char** argv) {
  int port = 3307;
  std::string dataDir = "./warpstore-data";
  bool sync = false;  // async durability by default (SingleStore's default); --sync-durability: every commit waits for the disk
  int partitions = 8;
  double slowMsArg = 1000;
  int httpPort = 0;
  std::string replicaOf;
  std::vector<std::string> leaves;
  std::string planCacheDir;
  bool planCacheDirSet = false;
  std::string bindAddr = "127.0.0.1", rootPw, tlsCert, tlsKey;
  bool insecure = false, noTls = false;
  double memLimitGb = -1;  // host memory for table data (segment columns); default half the RAM, 0: no limit
  if (const char* v = envSetting("MEMORY_LIMIT")) memLimitGb = atof(v);
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) port = atoi(argv[++i]);
    else if (a == "--data-dir" && i + 1 < argc) dataDir = argv[++i];
    else if (a == "--no-wal") dataDir.clear();
    else if (a == "--compress") g_compress = 1;
    else if (a == "--no-compress") g_compress = 0;
    else if (a == "--sync" || a == "--sync-durability") sync = true;
    else if (a == "--wal-archive" && i + 1 < argc) gWalArchive = argv[++i];
    else if (a == "--slow-query-ms" && i + 1 < argc) slowMsArg = atof(argv[++i]);
    else if (a == "--slow-log" && i + 1 < argc) monitor::slowPath = argv[++i];
    else if (a == "--sync-replicas" && i + 1 < argc) gSyncReplicas = atoi(argv[++i]);
    else if (a == "--sync-replica-timeout-ms" && i + 1 < argc) gSyncReplicaTimeoutMs = atoi(argv[++i]);
    else if (a == "--failover-timeout" && i + 1 < argc) gFailoverTimeoutS = atof(argv[++i]);
    else if (a == "--async-commit" || a == "--no-sync") sync = false;
    else if (a == "--partitions" && i + 1 < argc) partitions = atoi(argv[++i]);
    else if (a == "--plancache-dir" && i + 1 < argc) planCacheDir = argv[++i], planCacheDirSet = true;
    else if (a == "--vram-limit" && i + 1 < argc) gpuSetVramLimit((int64_t)(atof(argv[++i]) * (1ll << 30)));
    else if (a == "--memory-limit" && i + 1 < argc) memLimitGb = atof(argv[++i]);
    else if (a == "--http-port" && i + 1 < argc) httpPort = atoi(argv[++i]);
    else if (a == "--data-port" && i + 1 < argc) gDataPort = atoi(argv[++i]);
    else if (a == "--replica-of" && i + 1 < argc) replicaOf = argv[++i];
    else if (a == "--redundancy" && i + 1 < argc) g_redundancy = atoi(argv[++i]);  // 2: every leaf's shard mirrored on the next leaf
    else if (a == "--leaves" && i + 1 < argc) {  // cluster aggregator: comma-separated leaf HTTP URLs
      std::string l = argv[++i];
      for (size_t b = 0; b <= l.size();) {
        size_t e = l.find(',', b);
        if (e == std::string::npos) e = l.size();
        if (e > b) leaves.push_back(l.substr(b, e - b));
        b = e + 1;
      }
    }
    else if (a == "--codegen" && i + 1 < argc) {
      std::string m = argv[++i];
      jitSetDefaultMode(m == "off" ? JIT_OFF : m == "sync" ? JIT_SYNC : JIT_ON);
    }
    else if (a == "-v" || a == "--verbose") g_verbose = true;
    else if (a == "--bind" && i + 1 < argc) bindAddr = argv[++i];
    else if (a == "--root-password" && i + 1 < argc) rootPw = argv[++i];
    else if (a == "--tls-cert" && i + 1 < argc) tlsCert = argv[++i];
    else if (a == "--tls-key" && i + 1 < argc) tlsKey = argv[++i];
    else if (a == "--no-tls") noTls = true;
    else if (a == "--require-secure-transport") g_requireTls = true;
    else if (a == "--max-connections" && i + 1 < argc) g_maxConns = atoi(argv[++i]);
    else if (a == "--insecure-no-root-password") insecure = true;
    else {
      fprintf(stderr, "usage: %s [--port N] [--data-dir DIR | --no-wal] [--sync-durability] [--wal-archive DIR] [--slow-query-ms N] [--slow-log FILE] [--partitions N] [--codegen on|sync|off] [--plancache-dir DIR|\"\"] [--vram-limit GB] [--memory-limit GB] [--compress | --no-compress] [--http-port N] [--data-port N] [--replica-of http://user:pw@primary:httpport [--failover-timeout S]] [--sync-replicas N] [--leaves http://leaf1:httpport,..] [--redundancy 2] [--bind ADDR] [--root-password PW] [--tls-cert F --tls-key F | --no-tls] [--require-secure-transport] [--max-connections N] [--insecure-no-root-password] [--sync-replica-timeout-ms N] [-v]\n", argv[0]);
      return 2;
    }
  }
  signal(SIGPIPE, SIG_IGN);
  try {
    gpuInit();
  } catch (std::exception& e) {
    fprintf(stderr, "GPU init failed: %s\n", e.what());
    return 1;
  }
  fprintf(stderr, "[warpstore] GPU: %s\n", gpuDescribe().c_str());
  if (!dataDir.empty()) mkdir(dataDir.c_str(), 0755);
  g_dataDir = dataDir;
  // Compiled plans persist next to the data, or in ~/.cache/warpstore/plancache without a data dir.
  if (!planCacheDirSet) {
    const char* home = getenv("HOME");
    planCacheDir = !dataDir.empty() ? dataDir + "/plancache" : home ? std::string(home) + "/.cache/warpstore/plancache" : "";
  }
  jitSetCacheDir(planCacheDir);
  if (!planCacheDir.empty()) fprintf(stderr, "[warpstore] persistent plan cache: %s\n", planCacheDir.c_str());
  // Slow query log: <data dir>/slow.log, statements over 1 s (--slow-query-ms, SET GLOBAL long_query_time).
  if (monitor::slowPath.empty() && !dataDir.empty()) monitor::slowPath = dataDir + "/slow.log";
  monitor::slowMs = monitor::slowPath.empty() ? -1 : slowMsArg;
  {
    int64_t lim = -1;  // (no limit without a data dir: nothing to evict to)
    if (memLimitGb > 0) lim = (int64_t)(memLimitGb * (1ll << 30));
    else if (memLimitGb < 0) lim = (int64_t)sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE) / 2;
    tierInit(dataDir, lim);
    if (!dataDir.empty() && lim > 0)
      fprintf(stderr, "[warpstore] table data in host memory up to %.1f GB, the rest in %s/segments (--memory-limit)\n", lim / 1073741824.0, dataDir.c_str());
    if (!dataDir.empty())
      fprintf(stderr, "[warpstore] segment files: decompression %s; reads: %s\n", nvcompStatus().c_str(), gdsStatus().c_str());
  }
  Engine engine(dataDir, sync, partitions);
  fprintf(stderr, "[warpstore] %d partitions per sharded table\n", partitions);
  engine.recover();
  if (rootPw.empty())
    if (const char* e = envSetting("ROOT_PASSWORD")) rootPw = e;
  if (!rootPw.empty()) {
    std::string q;
    for (char ch : rootPw) q += ch == '\'' ? std::string("''") : ch == '\\' ? std::string("\\\\") : std::string(1, ch);
    engine.execute("ALTER USER 'root' IDENTIFIED BY '" + q + "'");
  }
  const bool loopbackOnly = isLoopback(bindAddr);
  if (engine.checkPassword("root", "")) {
    if (!loopbackOnly && !insecure) {
      fprintf(stderr, "[warpstore] refusing to listen on %s: root has no password. Set one with --root-password / WARPSTORE_ROOT_PASSWORD "
                      "(or pass --insecure-no-root-password)\n", bindAddr.c_str());
      return 1;
    }
    fprintf(stderr, "[warpstore] warning: root has no password%s\n", loopbackOnly ? " (listening on loopback only)" : "");
  }
  if (!noTls) {
    if (tlsInit(tlsCert, tlsKey, dataDir)) fprintf(stderr, "[warpstore] TLS enabled%s\n", g_requireTls ? " (required for remote clients)" : "");
    else if (!tlsCert.empty()) { fprintf(stderr, "[warpstore] cannot load TLS certificate %s / key %s\n", tlsCert.c_str(), tlsKey.c_str()); return 1; }
  }
  in_addr bindIn{};
  if (inet_pton(AF_INET, bindAddr == "localhost" ? "127.0.0.1" : bindAddr.c_str(), &bindIn) != 1) {
    fprintf(stderr, "[warpstore] --bind: not an IPv4 address: %s\n", bindAddr.c_str());
    return 1;
  }
  if (!leaves.empty()) {
    engine.setLeaves(leaves);
    fprintf(stderr, "[warpstore] cluster aggregator over %zu leaves\n", leaves.size());
  }
  if (!replicaOf.empty()) {
    fprintf(stderr, "[warpstore] read-only replica of %s\n", replicaOf.c_str());
    engine.startReplica(replicaOf);
  }

  int s = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)port);
  addr.sin_addr = bindIn;
  if (bind(s, (sockaddr*)&addr, sizeof addr) < 0 || listen(s, 128) < 0) {
    perror("bind/listen");
    return 1;
  }
  fprintf(stderr, "[warpstore] listening on %s:%d (MySQL protocol)%s\n", bindAddr.c_str(), port,
          dataDir.empty() ? ", WAL disabled" : (", WAL in " + dataDir).c_str());
  if (httpPort > 0) {  // HTTP Data API
    int hs = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(hs, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in ha{};
    ha.sin_family = AF_INET;
    ha.sin_port = htons((uint16_t)httpPort);
    ha.sin_addr = bindIn;
    if (bind(hs, (sockaddr*)&ha, sizeof ha) < 0 || listen(hs, 128) < 0) {
      perror("http bind/listen");
      return 1;
    }
    fprintf(stderr, "[warpstore] HTTP Data API on port %d (POST /api/v2/query/rows, /api/v2/query/tuples, /api/v2/exec)\n", httpPort);
    std::thread([hs, &engine] {
      for (;;) {
        int fd = accept(hs, nullptr, nullptr);
        if (fd >= 0) std::thread(serveHttp, fd, &engine).detach();
      }
    }).detach();
  }
  if (httpPort > 0 && gDataPort == 0 && httpPort + 10000 < 65536) gDataPort = httpPort + 10000;
  if (gDataPort < 0) gDataPort = 0;
  if (gDataPort > 0) {  // cluster data plane
    int ds = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(ds, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in da{};
    da.sin_family = AF_INET;
    da.sin_port = htons((uint16_t)gDataPort);
    da.sin_addr = bindIn;
    if (bind(ds, (sockaddr*)&da, sizeof da) < 0 || listen(ds, 128) < 0) {
      fprintf(stderr, "[warpstore] data plane: cannot listen on port %d (cluster traffic uses HTTP)\n", gDataPort);
      close(ds);
      gDataPort = 0;
    } else {
      fprintf(stderr, "[warpstore] cluster data plane on port %d\n", gDataPort);
      std::thread([ds, &engine] {
        for (;;) {
          int fd = accept(ds, nullptr, nullptr);
          if (fd >= 0) std::thread(serveData, fd, &engine).detach();
        }
      }).detach();
    }
  }
  for (;;) {
    int fd = accept(s, nullptr, nullptr);
    if (fd < 0) continue;
    std::thread(serve, fd, &engine).detach();
  }
}
