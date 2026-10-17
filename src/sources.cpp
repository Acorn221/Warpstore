#include "sources.h"

#include <arpa/inet.h>
#include <curl/curl.h>
#include <fnmatch.h>
#include <lz4frame.h>
#include <netdb.h>
#include <snappy.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>
#include <zlib.h>
#include <zstd.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "common.h"

namespace gdb {

// ---- JSON helpers -------------------------------------------------------------------------
namespace {

void skipWs(const std::string& s, size_t& i) {
  while (i < s.size() && isspace((unsigned char)s[i])) ++i;
}
// Parses a JSON string at s[i] == '"'; returns the unescaped text.
std::string jstr(const std::string& s, size_t& i) {
  std::string o;
  ++i;
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\' && i + 1 < s.size()) {
      char c = s[++i];
      switch (c) {
        case 'n': o += '\n'; break;
        case 't': o += '\t'; break;
        case 'r': o += '\r'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'u': {
          unsigned cp = i + 4 < s.size() ? (unsigned)std::stoul(s.substr(i + 1, 4), nullptr, 16) : '?';
          i += 4;
          if (cp < 0x80) o += (char)cp;
          else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
          else { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
          break;
        }
        default: o += c;
      }
      ++i;
      continue;
    }
    o += s[i++];
  }
  ++i;
  return o;
}
// Skips any JSON value; returns its raw text.
std::string jraw(const std::string& s, size_t& i) {
  size_t b = i;
  if (s[i] == '"') {
    jstr(s, i);
    return s.substr(b, i - b);
  }
  if (s[i] == '{' || s[i] == '[') {
    int depth = 0;
    for (; i < s.size(); ++i) {
      if (s[i] == '"') { jstr(s, i); --i; continue; }
      if (s[i] == '{' || s[i] == '[') ++depth;
      if (s[i] == '}' || s[i] == ']') {
        if (--depth == 0) { ++i; break; }
      }
    }
    return s.substr(b, i - b);
  }
  while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' && !isspace((unsigned char)s[i])) ++i;
  return s.substr(b, i - b);
}
// Top-level fields of a JSON object: key -> (value text, is null). Strings are unescaped.
std::map<std::string, std::pair<std::string, bool>> jobject(const std::string& s) {
  std::map<std::string, std::pair<std::string, bool>> out;
  size_t i = 0;
  skipWs(s, i);
  if (i >= s.size() || s[i] != '{') return out;
  ++i;
  for (;;) {
    skipWs(s, i);
    if (i >= s.size() || s[i] == '}') break;
    if (s[i] != '"') break;
    std::string k = jstr(s, i);
    skipWs(s, i);
    if (i < s.size() && s[i] == ':') ++i;
    skipWs(s, i);
    if (i >= s.size()) break;
    if (s[i] == '"') out[k] = {jstr(s, i), false};
    else {
      std::string v = jraw(s, i);
      out[k] = {v, v == "null"};
      if (v == "true") out[k].first = "1";
      if (v == "false") out[k].first = "0";
    }
    skipWs(s, i);
    if (i < s.size() && s[i] == ',') ++i;
  }
  return out;
}
std::string lowerS(std::string s) {
  for (auto& c : s) c = (char)tolower((unsigned char)c);
  return s;
}

}  // namespace

std::string jsonField(const std::string& json, const std::string& key) {
  auto o = jobject(json);
  auto it = o.find(key);
  return it == o.end() || it->second.second ? "" : it->second.first;
}

std::string jsonLinesToTsv(const std::string& data, const std::vector<std::string>& cols) {
  std::string out;
  size_t a = 0;
  while (a < data.size()) {
    size_t b = data.find('\n', a);
    if (b == std::string::npos) b = data.size();
    std::string line = data.substr(a, b - a);
    a = b + 1;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    auto o = jobject(line);
    for (size_t c = 0; c < cols.size(); ++c) {
      if (c) out += '\t';
      // a::b::c: nested keys (case-insensitive)
      std::string v;
      bool found = true;
      auto cur = o;
      std::string path = cols[c];
      for (;;) {
        size_t sep = path.find("::");
        std::string k = lowerS(path.substr(0, sep));
        auto it = cur.begin();
        for (; it != cur.end(); ++it)
          if (lowerS(it->first) == k) break;
        if (it == cur.end() || it->second.second) { found = false; break; }
        if (sep == std::string::npos) { v = it->second.first; break; }
        auto inner = jobject(it->second.first);
        cur = inner;
        path = path.substr(sep + 2);
      }
      if (!found) {
        out += "\\N";
        continue;
      }
      for (auto& ch : v)
        if (ch == '\t' || ch == '\n' || ch == '\r') ch = ' ';
      if (!v.empty() && v[0] == '"') {
        std::string q = "\"";
        for (char ch : v) q += ch == '"' ? std::string("\"\"") : std::string(1, ch);
        v = q + "\"";
      }
      out += v;
    }
    out += '\n';
  }
  return out;
}

// ---- Decompression ------------------------------------------------------------------------
namespace {

std::string gunzip(const std::string& in) {
  z_stream zs{};
  if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK) throw SqlError("zlib init failed", 1105);
  zs.next_in = (Bytef*)in.data();
  zs.avail_in = (uInt)in.size();
  std::string out;
  char buf[1 << 16];
  int rc;
  do {
    zs.next_out = (Bytef*)buf;
    zs.avail_out = sizeof buf;
    rc = inflate(&zs, Z_NO_FLUSH);
    if (rc != Z_OK && rc != Z_STREAM_END) {
      inflateEnd(&zs);
      throw SqlError("gzip data is corrupt", 1105);
    }
    out.append(buf, sizeof buf - zs.avail_out);
    if (rc == Z_STREAM_END && zs.avail_in > 0) inflateReset(&zs);  // concatenated members
  } while (zs.avail_in > 0 || rc != Z_STREAM_END);
  inflateEnd(&zs);
  return out;
}

std::string unzstd(const std::string& in) {
  std::unique_ptr<ZSTD_DCtx, size_t (*)(ZSTD_DCtx*)> d(ZSTD_createDCtx(), ZSTD_freeDCtx);
  ZSTD_inBuffer ib{in.data(), in.size(), 0};
  std::string out;
  std::vector<char> buf(ZSTD_DStreamOutSize());
  while (ib.pos < ib.size) {
    ZSTD_outBuffer ob{buf.data(), buf.size(), 0};
    size_t rc = ZSTD_decompressStream(d.get(), &ob, &ib);
    if (ZSTD_isError(rc)) throw SqlError(std::string("zstd: ") + ZSTD_getErrorName(rc), 1105);
    out.append(buf.data(), ob.pos);
    if (rc == 0 && ob.pos == 0 && ib.pos < ib.size) continue;
  }
  return out;
}

std::string unlz4(const std::string& in) {
  LZ4F_dctx* d;
  if (LZ4F_isError(LZ4F_createDecompressionContext(&d, LZ4F_VERSION))) throw SqlError("lz4 init failed", 1105);
  std::string out;
  std::vector<char> buf(1 << 16);
  size_t pos = 0;
  while (pos < in.size()) {
    size_t outSz = buf.size(), inSz = in.size() - pos;
    size_t rc = LZ4F_decompress(d, buf.data(), &outSz, in.data() + pos, &inSz, nullptr);
    if (LZ4F_isError(rc)) {
      LZ4F_freeDecompressionContext(d);
      throw SqlError(std::string("lz4: ") + LZ4F_getErrorName(rc), 1105);
    }
    out.append(buf.data(), outSz);
    pos += inSz;
    if (inSz == 0 && outSz == 0) break;
  }
  LZ4F_freeDecompressionContext(d);
  return out;
}

std::string unsnappyKafka(const std::string& in) {
  // Kafka's Java client uses xerial framing: magic, version ints, then [int32 len][block]*.
  static const char magic[8] = {(char)0x82, 'S', 'N', 'A', 'P', 'P', 'Y', 0};
  std::string out;
  if (in.size() >= 16 && memcmp(in.data(), magic, 8) == 0) {
    size_t pos = 16;
    while (pos + 4 <= in.size()) {
      uint32_t n = ntohl(*(const uint32_t*)(in.data() + pos));
      pos += 4;
      std::string blk;
      if (pos + n > in.size() || !snappy::Uncompress(in.data() + pos, n, &blk)) throw SqlError("snappy data is corrupt", 1105);
      out += blk;
      pos += n;
    }
    return out;
  }
  if (!snappy::Uncompress(in.data(), in.size(), &out)) throw SqlError("snappy data is corrupt", 1105);
  return out;
}

bool endsWith(const std::string& s, const char* suf) {
  size_t n = strlen(suf);
  return s.size() >= n && strcasecmp(s.c_str() + s.size() - n, suf) == 0;
}

}  // namespace

std::string decompressByName(const std::string& name, std::string data) {
  if (endsWith(name, ".gz") || endsWith(name, ".gzip")) return gunzip(data);
  if (endsWith(name, ".zst") || endsWith(name, ".zstd")) return unzstd(data);
  if (endsWith(name, ".lz4")) return unlz4(data);
  return data;
}

// ---- HTTP / S3 ----------------------------------------------------------------------------
namespace {

size_t sink(char* p, size_t sz, size_t n, void* ud) {
  static_cast<std::string*>(ud)->append(p, sz * n);
  return sz * n;
}

std::string curlGet(const std::string& url, const std::string& region, const std::string& key, const std::string& secret,
                    const std::string& token) {
  static std::once_flag once;
  std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
  std::unique_ptr<CURL, void (*)(CURL*)> c(curl_easy_init(), curl_easy_cleanup);
  std::string body;
  curl_easy_setopt(c.get(), CURLOPT_URL, url.c_str());
  curl_easy_setopt(c.get(), CURLOPT_WRITEFUNCTION, sink);
  curl_easy_setopt(c.get(), CURLOPT_WRITEDATA, &body);
  curl_easy_setopt(c.get(), CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(c.get(), CURLOPT_CONNECTTIMEOUT, 10L);
  curl_easy_setopt(c.get(), CURLOPT_TIMEOUT, 600L);
  curl_slist* hdrs = nullptr;
  std::string userpwd, sigv4;
  if (!key.empty()) {
    sigv4 = "aws:amz:" + region + ":s3";
    userpwd = key + ":" + secret;
    curl_easy_setopt(c.get(), CURLOPT_AWS_SIGV4, sigv4.c_str());
    curl_easy_setopt(c.get(), CURLOPT_USERPWD, userpwd.c_str());
    hdrs = curl_slist_append(hdrs, "x-amz-content-sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    if (!token.empty()) hdrs = curl_slist_append(hdrs, ("x-amz-security-token: " + token).c_str());
    curl_easy_setopt(c.get(), CURLOPT_HTTPHEADER, hdrs);
  }
  CURLcode rc = curl_easy_perform(c.get());
  long status = 0;
  curl_easy_getinfo(c.get(), CURLINFO_RESPONSE_CODE, &status);
  if (hdrs) curl_slist_free_all(hdrs);
  if (rc != CURLE_OK) throw SqlError("HTTP request to " + url + " failed: " + curl_easy_strerror(rc), 1105);
  if (status >= 300) {
    std::string msg = body;
    size_t a = msg.find("<Message>"), b = msg.find("</Message>");
    if (a != std::string::npos && b != std::string::npos) msg = msg.substr(a + 9, b - a - 9);
    if (msg.size() > 200) msg = msg.substr(0, 200);
    throw SqlError("HTTP " + std::to_string(status) + " from " + url + ": " + msg, 1105);
  }
  return body;
}

std::string esc(const std::string& s, bool slash) {
  static const char* hex = "0123456789ABCDEF";
  std::string o;
  for (unsigned char c : s) {
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || (c == '/' && !slash)) o += (char)c;
    else {
      o += '%';
      o += hex[c >> 4];
      o += hex[c & 15];
    }
  }
  return o;
}

std::string xmlUnescape(std::string s) {
  static const std::pair<const char*, const char*> ents[] = {{"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}};
  for (auto& [e, r] : ents)
    for (size_t p; (p = s.find(e)) != std::string::npos;) s.replace(p, strlen(e), r);
  return s;
}

struct S3Ctx {
  std::string region, endpoint, key, secret, token;
  S3Ctx(const std::string& config, const std::string& creds) {
    region = jsonField(config, "region");
    if (region.empty()) region = "us-east-1";
    endpoint = jsonField(config, "endpoint_url");
    while (!endpoint.empty() && endpoint.back() == '/') endpoint.pop_back();
    key = jsonField(creds, "aws_access_key_id");
    secret = jsonField(creds, "aws_secret_access_key");
    token = jsonField(creds, "aws_session_token");
  }
  std::string base(const std::string& bucket) const {
    return endpoint.empty() ? "https://" + bucket + ".s3." + region + ".amazonaws.com" : endpoint + "/" + bucket;
  }
};

}  // namespace

std::string httpGet(const std::string& url) { return curlGet(url, "", "", "", ""); }

std::string httpPost(const std::string& url, const std::string& body, const std::string& contentType) {
  static std::once_flag once;
  std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
  thread_local std::unique_ptr<CURL, void (*)(CURL*)> c(curl_easy_init(), curl_easy_cleanup);  // keep-alive per thread
  curl_easy_reset(c.get());
  std::string out;
  curl_slist* hdrs = curl_slist_append(nullptr, ("Content-Type: " + contentType).c_str());
  hdrs = curl_slist_append(hdrs, "Expect:");  // no 100-continue round trip (curl waits up to 1 s for it)
  curl_easy_setopt(c.get(), CURLOPT_URL, url.c_str());
  curl_easy_setopt(c.get(), CURLOPT_POST, 1L);
  curl_easy_setopt(c.get(), CURLOPT_POSTFIELDS, body.data());
  curl_easy_setopt(c.get(), CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)body.size());
  curl_easy_setopt(c.get(), CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(c.get(), CURLOPT_WRITEFUNCTION, sink);
  curl_easy_setopt(c.get(), CURLOPT_WRITEDATA, &out);
  curl_easy_setopt(c.get(), CURLOPT_CONNECTTIMEOUT, 10L);
  curl_easy_setopt(c.get(), CURLOPT_TIMEOUT, 3600L);
  CURLcode rc = curl_easy_perform(c.get());
  long status = 0;
  curl_easy_getinfo(c.get(), CURLINFO_RESPONSE_CODE, &status);
  curl_slist_free_all(hdrs);
  if (rc != CURLE_OK) throw SqlError("HTTP request to " + url + " failed: " + curl_easy_strerror(rc), 1105);
  if (status >= 300) {
    std::string msg = jsonField(out, "error");
    throw SqlError(msg.empty() ? "HTTP " + std::to_string(status) + " from " + url : msg, 1105);
  }
  return out;
}

std::string s3Bucket(const std::string& path) {
  size_t p = path.find('/');
  return p == std::string::npos ? path : path.substr(0, p);
}

std::vector<S3Object> s3List(const std::string& path, const std::string& config, const std::string& creds) {
  S3Ctx c(config, creds);
  std::string bucket = s3Bucket(path);
  std::string pattern = path.size() > bucket.size() ? path.substr(bucket.size() + 1) : "";
  std::string prefix = pattern.substr(0, pattern.find_first_of("*?["));
  const bool glob = prefix.size() != pattern.size();
  std::vector<S3Object> out;
  std::string cont;
  for (int page = 0; page < 10000; ++page) {
    std::string url = c.base(bucket) + "/?list-type=2&prefix=" + esc(prefix, true);
    if (!cont.empty()) url += "&continuation-token=" + esc(cont, true);
    std::string xml = curlGet(url, c.region, c.key, c.secret, c.token);
    for (size_t p = 0; (p = xml.find("<Contents>", p)) != std::string::npos;) {
      size_t e = xml.find("</Contents>", p);
      std::string blk = xml.substr(p, e - p);
      p = e;
      auto tag = [&](const char* t) {
        std::string o = std::string("<") + t + ">", cl = std::string("</") + t + ">";
        size_t a = blk.find(o), b = blk.find(cl);
        return a == std::string::npos || b == std::string::npos ? std::string() : blk.substr(a + o.size(), b - a - o.size());
      };
      S3Object ob;
      ob.key = xmlUnescape(tag("Key"));
      ob.size = std::atoll(tag("Size").c_str());
      if (ob.key.empty() || ob.key.back() == '/') continue;
      if (glob && fnmatch(pattern.c_str(), ob.key.c_str(), 0) != 0) continue;
      out.push_back(ob);
    }
    size_t a = xml.find("<NextContinuationToken>");
    if (xml.find("<IsTruncated>true</IsTruncated>") == std::string::npos || a == std::string::npos) break;
    size_t b = xml.find("</NextContinuationToken>", a);
    cont = xmlUnescape(xml.substr(a + 23, b - a - 23));
  }
  return out;
}

std::string s3Get(const std::string& bucket, const std::string& key, const std::string& config, const std::string& creds) {
  S3Ctx c(config, creds);
  return curlGet(c.base(bucket) + "/" + esc(key, false), c.region, c.key, c.secret, c.token);
}

// ---- Kafka (plaintext wire protocol: Metadata v1, ListOffsets v1, Fetch v4, RecordBatch v2) ----
namespace {

struct Buf {
  std::string b;
  void i8(int8_t v) { b += (char)v; }
  void i16(int16_t v) { uint16_t x = htons((uint16_t)v); b.append((char*)&x, 2); }
  void i32(int32_t v) { uint32_t x = htonl((uint32_t)v); b.append((char*)&x, 4); }
  void i64(int64_t v) { i32((int32_t)(v >> 32)); i32((int32_t)(v & 0xffffffff)); }
  void str(const std::string& s) { i16((int16_t)s.size()); b += s; }
};

struct Rd {
  const std::string& b;
  size_t p = 0;
  explicit Rd(const std::string& b, size_t p = 0) : b(b), p(p) {}
  void need(size_t n) { if (p + n > b.size()) throw SqlError("truncated Kafka response", 1105); }
  int8_t i8() { need(1); return (int8_t)b[p++]; }
  int16_t i16() { need(2); uint16_t x; memcpy(&x, b.data() + p, 2); p += 2; return (int16_t)ntohs(x); }
  int32_t i32() { need(4); uint32_t x; memcpy(&x, b.data() + p, 4); p += 4; return (int32_t)ntohl(x); }
  int64_t i64() { int64_t hi = (uint32_t)i32(); int64_t lo = (uint32_t)i32(); return (hi << 32) | lo; }
  std::string str() { int16_t n = i16(); if (n < 0) return ""; need(n); std::string s = b.substr(p, n); p += n; return s; }
  int64_t varint() {
    uint64_t v = 0;
    for (int sh = 0; sh < 64; sh += 7) {
      need(1);
      uint8_t c = (uint8_t)b[p++];
      v |= (uint64_t)(c & 0x7f) << sh;
      if (!(c & 0x80)) break;
    }
    return (int64_t)(v >> 1) ^ -(int64_t)(v & 1);
  }
};

class Conn {
 public:
  Conn(const std::string& host, int port) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res)
      throw SqlError("cannot resolve Kafka broker " + host, 1105);
    fd_ = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    timeval tv{15, 0};
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    if (fd_ < 0 || connect(fd_, res->ai_addr, res->ai_addrlen) != 0) {
      freeaddrinfo(res);
      if (fd_ >= 0) close(fd_);
      throw SqlError("cannot connect to Kafka broker " + host + ":" + std::to_string(port), 1105);
    }
    freeaddrinfo(res);
  }
  ~Conn() { if (fd_ >= 0) close(fd_); }
  std::string call(int16_t api, int16_t ver, const std::string& body) {
    Buf h;
    h.i16(api);
    h.i16(ver);
    h.i32(++corr_);
    h.str("gpudb");
    std::string msg = h.b + body;
    Buf sz;
    sz.i32((int32_t)msg.size());
    msg = sz.b + msg;
    for (size_t off = 0; off < msg.size();) {
      ssize_t n = ::send(fd_, msg.data() + off, msg.size() - off, MSG_NOSIGNAL);
      if (n <= 0) throw SqlError("Kafka send failed", 1105);
      off += n;
    }
    std::string len = readN(4);
    int32_t n = (int32_t)ntohl(*(const uint32_t*)len.data());
    std::string resp = readN(n);
    return resp.substr(4);  // correlation id
  }

 private:
  std::string readN(size_t n) {
    std::string o(n, '\0');
    for (size_t off = 0; off < n;) {
      ssize_t r = ::recv(fd_, o.data() + off, n - off, 0);
      if (r <= 0) throw SqlError("Kafka broker closed the connection", 1105);
      off += r;
    }
    return o;
  }
  int fd_ = -1;
  int32_t corr_ = 0;
};

std::pair<std::string, int> hostPort(const std::string& s) {
  size_t c = s.rfind(':');
  if (c == std::string::npos) return {s, 9092};
  return {s.substr(0, c), std::atoi(s.c_str() + c + 1)};
}

void parseRecords(const std::string& recs, int part, int64_t from, std::vector<KafkaMsg>& out, int64_t& next) {
  size_t p = 0;
  while (p + 12 <= recs.size()) {
    Rd r(recs, p);
    int64_t base = r.i64();
    int32_t blen = r.i32();
    if (p + 12 + (size_t)blen > recs.size()) break;  // partial batch at the end of the fetch
    size_t end = p + 12 + blen;
    r.i32();  // leader epoch
    int8_t magic = r.i8();
    if (magic != 2) throw SqlError("Kafka message format v" + std::to_string(magic) + " is not supported (need v2)", 1235);
    r.i32();  // crc
    int16_t attrs = r.i16();
    int32_t lastDelta = r.i32();
    r.i64();
    r.i64();
    r.i64();
    r.i16();
    r.i32();
    int32_t count = r.i32();
    next = std::max(next, base + lastDelta + 1);
    if (attrs & 0x20) {  // control batch (transaction markers)
      p = end;
      continue;
    }
    std::string body = recs.substr(r.p, end - r.p);
    switch (attrs & 7) {
      case 0: break;
      case 1: body = gunzip(body); break;
      case 2: body = unsnappyKafka(body); break;
      case 3: body = unlz4(body); break;
      case 4: body = unzstd(body); break;
      default: throw SqlError("unknown Kafka compression codec", 1105);
    }
    Rd rr(body);
    for (int32_t i = 0; i < count && rr.p < body.size(); ++i) {
      int64_t len = rr.varint();
      size_t recEnd = rr.p + len;
      rr.i8();
      rr.varint();  // timestamp delta
      int64_t off = base + rr.varint();
      int64_t kl = rr.varint();
      if (kl > 0) rr.p += kl;
      int64_t vl = rr.varint();
      std::string v;
      if (vl > 0) {
        rr.need(vl);
        v = body.substr(rr.p, vl);
        rr.p += vl;
      }
      rr.p = recEnd;
      if (off >= from) out.push_back({part, off, std::move(v)});
    }
    p = end;
  }
}

}  // namespace

std::vector<KafkaMsg> kafkaPoll(const std::string& path, std::map<int, int64_t>& next, int32_t maxBytes) {
  size_t slash = path.find('/');
  if (slash == std::string::npos) throw SqlError("Kafka source must be 'host:port/topic'", 1235);
  std::string brokersS = path.substr(0, slash), topic = path.substr(slash + 1);
  std::vector<std::string> brokers;
  for (size_t a = 0; a <= brokersS.size();) {
    size_t b = brokersS.find(',', a);
    if (b == std::string::npos) b = brokersS.size();
    brokers.push_back(brokersS.substr(a, b - a));
    a = b + 1;
  }
  // Metadata: partitions and their leaders.
  std::map<int32_t, std::pair<std::string, int>> nodes;
  std::map<int32_t, int32_t> leaderOf;
  std::string lastErr;
  for (auto& bs : brokers) {
    try {
      auto [h, port] = hostPort(bs);
      Conn c(h, port);
      Buf q;
      q.i32(1);
      q.str(topic);
      std::string resp = c.call(3, 1, q.b);
      Rd r(resp);
      int32_t nb = r.i32();
      for (int i = 0; i < nb; ++i) {
        int32_t id = r.i32();
        std::string host = r.str();
        int32_t prt = r.i32();
        r.str();  // rack
        nodes[id] = {host, prt};
      }
      r.i32();  // controller
      int32_t nt = r.i32();
      for (int i = 0; i < nt; ++i) {
        int16_t err = r.i16();
        std::string name = r.str();
        r.i8();
        int32_t np = r.i32();
        if (err) throw SqlError("Kafka topic '" + name + "' error " + std::to_string(err), 1105);
        for (int j = 0; j < np; ++j) {
          r.i16();
          int32_t pid = r.i32(), leader = r.i32();
          int32_t nr = r.i32();
          for (int k = 0; k < nr; ++k) r.i32();
          int32_t ni = r.i32();
          for (int k = 0; k < ni; ++k) r.i32();
          leaderOf[pid] = leader;
        }
      }
      break;
    } catch (std::exception& e) {
      lastErr = e.what();
    }
  }
  if (leaderOf.empty()) throw SqlError("Kafka metadata failed: " + (lastErr.empty() ? "topic has no partitions" : lastErr), 1105);
  // Group partitions by leader; one connection per leader.
  std::map<int32_t, std::vector<int32_t>> byLeader;
  for (auto& [pid, l] : leaderOf) byLeader[l].push_back(pid);
  std::vector<KafkaMsg> out;
  for (auto& [leader, parts] : byLeader) {
    auto nit = nodes.find(leader);
    if (nit == nodes.end()) continue;
    Conn c(nit->second.first, nit->second.second);
    std::vector<int32_t> fresh;
    for (int32_t pid : parts)
      if (!next.count(pid)) fresh.push_back(pid);
    if (!fresh.empty()) {  // ListOffsets: earliest
      Buf q;
      q.i32(-1);
      q.i32(1);
      q.str(topic);
      q.i32((int32_t)fresh.size());
      for (int32_t pid : fresh) {
        q.i32(pid);
        q.i64(-2);
      }
      std::string resp = c.call(2, 1, q.b);
      Rd r(resp);
      int32_t nt = r.i32();
      for (int i = 0; i < nt; ++i) {
        r.str();
        int32_t np = r.i32();
        for (int j = 0; j < np; ++j) {
          int32_t pid = r.i32();
          int16_t err = r.i16();
          r.i64();
          int64_t off = r.i64();
          next[pid] = err ? 0 : off;
        }
      }
    }
    Buf q;  // Fetch v4
    q.i32(-1);
    q.i32(100);  // max wait ms
    q.i32(1);
    q.i32(maxBytes);
    q.i8(0);
    q.i32(1);
    q.str(topic);
    q.i32((int32_t)parts.size());
    for (int32_t pid : parts) {
      q.i32(pid);
      q.i64(next[pid]);
      q.i32(maxBytes);
    }
    std::string resp = c.call(1, 4, q.b);
    Rd r(resp);
    r.i32();  // throttle
    int32_t nt = r.i32();
    for (int i = 0; i < nt; ++i) {
      r.str();
      int32_t np = r.i32();
      for (int j = 0; j < np; ++j) {
        int32_t pid = r.i32();
        int16_t err = r.i16();
        r.i64();
        r.i64();
        int32_t na = r.i32();
        for (int k = 0; k < na; ++k) {
          r.i64();
          r.i64();
        }
        int32_t rl = r.i32();
        std::string recs;
        if (rl > 0) {
          r.need(rl);
          recs = resp.substr(r.p, rl);
          r.p += rl;
        }
        if (err == 1) {  // OFFSET_OUT_OF_RANGE: start over from the earliest offset next poll
          next.erase(pid);
          continue;
        }
        if (err) throw SqlError("Kafka fetch error " + std::to_string(err) + " on partition " + std::to_string(pid), 1105);
        int64_t nx = next[pid];
        parseRecords(recs, pid, next[pid], out, nx);
        next[pid] = nx;
      }
    }
  }
  return out;
}

}  // namespace gdb
