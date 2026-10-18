#include "mysqlcdc.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/sha.h>
#include <openssl/ssl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "common.h"
#include "engine.h"
#include "sql.h"
#include "storage.h"

namespace gdb {
namespace {

// ================================================================== MySQL client protocol

struct MyErr : SqlError {
  MyErr(const std::string& m, int c = 2013) : SqlError("MySQL source: " + m, c) {}
};

uint64_t lenenc(const std::string& b, size_t& i) {
  if (i >= b.size()) throw MyErr("truncated packet");
  const uint8_t c = (uint8_t)b[i++];
  auto le = [&](int n) {
    if (i + n > b.size()) throw MyErr("truncated packet");
    uint64_t v = 0;
    for (int k = 0; k < n; ++k) v |= (uint64_t)(uint8_t)b[i + k] << (8 * k);
    i += n;
    return v;
  };
  if (c < 0xfb) return c;
  if (c == 0xfc) return le(2);
  if (c == 0xfd) return le(3);
  if (c == 0xfe) return le(8);
  return ~0ull;  // 0xfb: NULL
}

std::string sha1s(const std::string& x) {
  unsigned char d[20];
  SHA1((const unsigned char*)x.data(), x.size(), d);
  return std::string((const char*)d, 20);
}
std::string sha256s(const std::string& x) {
  unsigned char d[32];
  SHA256((const unsigned char*)x.data(), x.size(), d);
  return std::string((const char*)d, 32);
}
std::string xorStr(const std::string& a, const std::string& b) {
  std::string o = a;
  for (size_t i = 0; i < o.size(); ++i) o[i] ^= b[i % b.size()];
  return o;
}

class MyConn {
 public:
  MyConn(const std::string& host, int port, const std::string& user, const std::string& pw, const std::string& db, bool tls)
      : user_(user), pw_(pw) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) throw MyErr("cannot resolve " + host, 2005);
    fd_ = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    timeval tv{30, 0};
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    const bool ok = fd_ >= 0 && ::connect(fd_, res->ai_addr, res->ai_addrlen) == 0;
    freeaddrinfo(res);
    if (!ok) {
      if (fd_ >= 0) ::close(fd_);
      fd_ = -1;
      throw MyErr("cannot connect to " + host + ":" + std::to_string(port), 2003);
    }
    int one = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    handshake(db, tls);
  }
  ~MyConn() {
    if (ssl_) {
      SSL_shutdown(ssl_);
      SSL_free(ssl_);
    }
    if (ctx_) SSL_CTX_free(ctx_);
    if (fd_ >= 0) ::close(fd_);
  }
  MyConn(const MyConn&) = delete;
  MyConn& operator=(const MyConn&) = delete;

  bool readable(int ms) {  // data waiting (or buffered by TLS) within ms
    if (ssl_ && SSL_pending(ssl_) > 0) return true;
    pollfd p{fd_, POLLIN, 0};
    return ::poll(&p, 1, ms) > 0;
  }

  // Text-protocol query: rows of cells (nullopt as a flag pair), columns names in cols.
  struct Result {
    std::vector<std::string> cols;
    std::vector<std::vector<std::string>> rows;
    std::vector<std::vector<bool>> nulls;
  };
  Result query(const std::string& sql) {
    Result r;
    queryStream(sql, r.cols, [&](std::vector<std::string>& row, std::vector<bool>& nul) {
      r.rows.push_back(row);
      r.nulls.push_back(nul);
    });
    return r;
  }
  void exec(const std::string& sql) { query(sql); }
  // Streams the rows of a result set to fn (one row at a time: big tables never sit in memory).
  void queryStream(const std::string& sql, std::vector<std::string>& cols,
                   const std::function<void(std::vector<std::string>&, std::vector<bool>&)>& fn) {
    seq_ = 0;
    writePacket(std::string(1, '\x03') + sql);
    std::string p = readPacket();
    if ((uint8_t)p[0] == 0xff) throw error(p);
    if ((uint8_t)p[0] == 0x00) return;  // OK: no result set
    size_t i = 0;
    const uint64_t n = lenenc(p, i);
    cols.clear();
    for (uint64_t c = 0; c < n; ++c) {
      std::string d = readPacket();
      size_t k = 0;
      std::string f[6];
      for (auto& x : f) {
        uint64_t l = lenenc(d, k);
        x = d.substr(k, l);
        k += l;
      }
      cols.push_back(f[4]);
    }
    p = readPacket();  // EOF after the column definitions
    std::vector<std::string> row(n);
    std::vector<bool> nul(n);
    for (;;) {
      p = readPacket();
      if ((uint8_t)p[0] == 0xfe && p.size() < 9) break;  // EOF
      if ((uint8_t)p[0] == 0xff) throw error(p);
      size_t k = 0;
      for (uint64_t c = 0; c < n; ++c) {
        if ((uint8_t)p[k] == 0xfb) {
          ++k;
          nul[c] = true;
          row[c].clear();
          continue;
        }
        uint64_t l = lenenc(p, k);
        nul[c] = false;
        row[c].assign(p, k, l);
        k += l;
      }
      fn(row, nul);
    }
  }

  // Binlog stream: COM_BINLOG_DUMP from file:pos; then readEvent() returns events (without the OK byte).
  void binlogDump(const std::string& file, uint32_t pos, uint32_t serverId) {
    seq_ = 0;
    std::string p(1, '\x12');
    put(p, pos, 4);
    put(p, 0, 2);  // flags: block
    put(p, serverId, 4);
    p += file;
    writePacket(p);
  }
  std::string readEvent() {
    std::string p = readPacket();
    if ((uint8_t)p[0] == 0xff) throw error(p);
    if ((uint8_t)p[0] == 0xfe && p.size() < 9) throw MyErr("binlog stream ended");
    return p.substr(1);
  }

 private:
  int fd_ = -1;
  SSL_CTX* ctx_ = nullptr;
  SSL* ssl_ = nullptr;
  uint8_t seq_ = 0;
  std::string user_, pw_;

  static void put(std::string& s, uint64_t v, int n) {
    for (int k = 0; k < n; ++k) s += (char)((v >> (8 * k)) & 0xff);
  }
  MyErr error(const std::string& p) {
    const int code = p.size() >= 3 ? (uint8_t)p[1] | ((uint8_t)p[2] << 8) : 2013;
    std::string msg = p.size() > 9 && p[3] == '#' ? p.substr(9) : p.substr(std::min<size_t>(3, p.size()));
    return MyErr(msg + " (error " + std::to_string(code) + ")", 1105);
  }
  void rawWrite(const char* d, size_t n) {
    while (n) {
      ssize_t k = ssl_ ? SSL_write(ssl_, d, (int)n) : ::send(fd_, d, n, MSG_NOSIGNAL);
      if (k <= 0) throw MyErr("connection lost (write)");
      d += k;
      n -= (size_t)k;
    }
  }
  void rawRead(char* d, size_t n) {
    while (n) {
      ssize_t k = ssl_ ? SSL_read(ssl_, d, (int)n) : ::recv(fd_, d, n, 0);
      if (k <= 0) throw MyErr("connection lost (read)");
      d += k;
      n -= (size_t)k;
    }
  }
  void writePacket(const std::string& payload) {
    size_t off = 0;
    do {
      const size_t n = std::min<size_t>(payload.size() - off, 0xffffff);
      char h[4] = {(char)(n & 0xff), (char)((n >> 8) & 0xff), (char)((n >> 16) & 0xff), (char)seq_++};
      rawWrite(h, 4);
      rawWrite(payload.data() + off, n);
      off += n;
      if (n < 0xffffff) break;
      if (off == payload.size()) {  // a payload of exactly k * 16 MB - 1 ends with an empty packet
        char e[4] = {0, 0, 0, (char)seq_++};
        rawWrite(e, 4);
        break;
      }
    } while (true);
  }
  std::string readPacket() {
    std::string out;
    for (;;) {
      unsigned char h[4];
      rawRead((char*)h, 4);
      const size_t n = h[0] | (h[1] << 8) | (h[2] << 16);
      seq_ = (uint8_t)(h[3] + 1);
      const size_t o = out.size();
      out.resize(o + n);
      if (n) rawRead(&out[o], n);
      if (n < 0xffffff) break;
    }
    if (out.empty()) throw MyErr("empty packet");
    return out;
  }
  std::string scramble(const std::string& plugin, const std::string& nonce) {
    if (pw_.empty()) return "";
    if (plugin == "caching_sha2_password") {
      const std::string a = sha256s(pw_);
      return xorStr(a, sha256s(sha256s(a) + nonce));
    }
    const std::string a = sha1s(pw_);  // mysql_native_password
    return xorStr(a, sha1s(nonce + sha1s(a)));
  }
  void handshake(const std::string& db, bool wantTls) {
    std::string p = readPacket();
    if ((uint8_t)p[0] == 0xff) throw error(p);
    size_t i = 1;
    while (i < p.size() && p[i]) ++i;  // server version
    i += 1 + 4;
    std::string nonce = p.substr(i, 8);
    i += 9;
    uint32_t caps = (uint8_t)p[i] | ((uint8_t)p[i + 1] << 8);
    i += 2 + 1 + 2;
    caps |= (uint32_t)((uint8_t)p[i] | ((uint8_t)p[i + 1] << 8)) << 16;
    i += 2;
    const int authLen = (uint8_t)p[i];
    i += 1 + 10;
    const int part2 = std::max(13, authLen - 8);
    nonce += p.substr(i, std::max(0, part2 - 1));  // (the last byte is a NUL)
    i += part2;
    std::string plugin;
    while (i < p.size() && p[i]) plugin += p[i++];
    if (plugin.empty()) plugin = "mysql_native_password";
    uint32_t mine = 0x1 | 0x4 | 0x200 | 0x2000 | 0x8000 | 0x20000 | 0x80000 | 0x200000;  // long pw, long flag, 4.1, txns, secure, multi-results, plugin auth, lenenc auth
    if (!db.empty()) mine |= 0x8;
    const bool tls = wantTls && (caps & 0x800);
    if (tls) {
      mine |= 0x800;
      std::string req;
      put(req, mine, 4);
      put(req, 1 << 24, 4);
      req += (char)45;  // utf8mb4_general_ci
      req += std::string(23, '\0');
      writePacket(req);
      ctx_ = SSL_CTX_new(TLS_client_method());
      SSL_CTX_set_verify(ctx_, SSL_VERIFY_NONE, nullptr);  // (MySQL's auto-generated certificates are self-signed)
      ssl_ = SSL_new(ctx_);
      SSL_set_fd(ssl_, fd_);
      if (SSL_connect(ssl_) != 1) throw MyErr("TLS handshake failed");
    }
    std::string resp;
    put(resp, mine, 4);
    put(resp, 1 << 24, 4);
    resp += (char)45;
    resp += std::string(23, '\0');
    resp += user_ + '\0';
    const std::string auth = scramble(plugin, nonce.substr(0, 20));
    resp += (char)auth.size();
    resp += auth;
    if (!db.empty()) resp += db + '\0';
    resp += plugin + '\0';
    writePacket(resp);
    for (int round = 0; round < 6; ++round) {
      std::string r = readPacket();
      const uint8_t k = (uint8_t)r[0];
      if (k == 0x00) return;  // OK
      if (k == 0xff) throw error(r);
      if (k == 0xfe) {  // auth switch: plugin name, new nonce
        size_t j = 1;
        plugin.clear();
        while (j < r.size() && r[j]) plugin += r[j++];
        nonce = r.substr(j + 1, 20);
        writePacket(scramble(plugin, nonce));
        continue;
      }
      if (k == 0x01 && r.size() >= 2 && r[1] == 0x03) continue;  // caching_sha2 fast auth: OK follows
      if (k == 0x01 && r.size() >= 2 && r[1] == 0x04) {          // caching_sha2 full auth
        if (ssl_) {
          writePacket(pw_ + '\0');
          continue;
        }
        writePacket(std::string(1, '\x02'));  // ask for the RSA public key
        std::string key = readPacket();
        if ((uint8_t)key[0] != 0x01) throw MyErr("expected the server's RSA key");
        BIO* bio = BIO_new_mem_buf(key.data() + 1, (int)key.size() - 1);
        EVP_PKEY* pk = PEM_read_bio_PUBKEY(bio, nullptr, nullptr, nullptr);
        BIO_free(bio);
        if (!pk) throw MyErr("bad RSA key from the server");
        const std::string plain = xorStr(pw_ + '\0', nonce.substr(0, 20));
        EVP_PKEY_CTX* pc = EVP_PKEY_CTX_new(pk, nullptr);
        size_t outLen = 0;
        std::string enc;
        if (EVP_PKEY_encrypt_init(pc) == 1 && EVP_PKEY_CTX_set_rsa_padding(pc, RSA_PKCS1_OAEP_PADDING) == 1 &&
            EVP_PKEY_encrypt(pc, nullptr, &outLen, (const unsigned char*)plain.data(), plain.size()) == 1) {
          enc.resize(outLen);
          if (EVP_PKEY_encrypt(pc, (unsigned char*)&enc[0], &outLen, (const unsigned char*)plain.data(), plain.size()) != 1) enc.clear();
          enc.resize(outLen);
        }
        EVP_PKEY_CTX_free(pc);
        EVP_PKEY_free(pk);
        if (enc.empty()) throw MyErr("RSA encryption failed");
        writePacket(enc);
        continue;
      }
      throw MyErr("unexpected authentication packet");
    }
    throw MyErr("authentication did not finish");
  }
};

// ================================================================== schema mapping

struct SrcCol {
  std::string name, dataType, columnType;
  bool isUnsigned = false;
  std::vector<std::string> enumLabels;  // ENUM / SET
  int precision = 0, scale = 0;
};
struct SrcTable {
  std::string name;
  std::vector<SrcCol> cols;
  std::vector<int> pk;  // indexes into cols
};

std::string quoteId(const std::string& s) {
  std::string o = "`";
  for (char c : s) o += c == '`' ? std::string("``") : std::string(1, c);
  return o + "`";
}
std::string sqlStr(const std::string& s) {  // warpstore / MySQL string literal (backslash escapes)
  std::string o = "'";
  for (char c : s) {
    switch (c) {
      case '\\': o += "\\\\"; break;
      case '\'': o += "\\'"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      case '\0': o += "\\0"; break;
      default: o += c;
    }
  }
  return o + "'";
}

std::vector<std::string> parseEnumLabels(const std::string& columnType) {  // enum('a','b''c') -> labels
  std::vector<std::string> out;
  size_t i = columnType.find('(');
  if (i == std::string::npos) return out;
  for (++i; i < columnType.size();) {
    if (columnType[i] != '\'') { ++i; continue; }
    std::string v;
    for (++i; i < columnType.size(); ++i) {
      if (columnType[i] == '\'' && i + 1 < columnType.size() && columnType[i + 1] == '\'') { v += '\''; ++i; }
      else if (columnType[i] == '\\' && i + 1 < columnType.size()) v += columnType[++i];
      else if (columnType[i] == '\'') { ++i; break; }
      else v += columnType[i];
    }
    out.push_back(v);
  }
  return out;
}

// warpstore column type for a MySQL column.
std::string targetType(const SrcCol& c) {
  const std::string& t = c.dataType;
  if (t == "tinyint" || t == "smallint" || t == "mediumint") return "INT";
  if (t == "int" || t == "integer") return c.isUnsigned ? "BIGINT" : "INT";
  if (t == "bigint" || t == "bit") return "BIGINT";
  if (t == "year") return "INT";
  if (t == "decimal" || t == "numeric")
    return c.precision <= 18 ? "DECIMAL(" + std::to_string(c.precision) + "," + std::to_string(c.scale) + ")" : "DOUBLE";
  if (t == "float" || t == "double" || t == "real") return "DOUBLE";
  if (t == "date") return "DATE";
  if (t == "datetime" || t == "timestamp") return "DATETIME";
  if (t == "json") return "JSON";
  return "VARCHAR(255)";  // char / varchar / text / enum / set / time / binary / blob / geometry
}

std::vector<SrcTable> loadSchema(MyConn& c, const std::string& db, const std::set<std::string>& only) {
  std::vector<SrcTable> out;
  auto tables = c.query("SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = " + sqlStr(db) +
                        " AND TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME");
  for (auto& r : tables.rows) {
    if (!only.empty() && !only.count(r[0])) continue;
    SrcTable t;
    t.name = r[0];
    auto cols = c.query("SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, COALESCE(NUMERIC_PRECISION, 0), COALESCE(NUMERIC_SCALE, 0) "
                        "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = " + sqlStr(db) + " AND TABLE_NAME = " + sqlStr(t.name) +
                        " ORDER BY ORDINAL_POSITION");
    for (auto& cr : cols.rows) {
      SrcCol sc;
      sc.name = cr[0];
      sc.dataType = lower(cr[1]);
      sc.columnType = lower(cr[2]);
      sc.isUnsigned = sc.columnType.find("unsigned") != std::string::npos;
      sc.precision = std::atoi(cr[3].c_str());
      sc.scale = std::atoi(cr[4].c_str());
      if (sc.dataType == "enum" || sc.dataType == "set") sc.enumLabels = parseEnumLabels(cr[2]);
      t.cols.push_back(sc);
    }
    auto pk = c.query("SELECT COLUMN_NAME FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA = " + sqlStr(db) +
                      " AND TABLE_NAME = " + sqlStr(t.name) + " AND CONSTRAINT_NAME = 'PRIMARY' ORDER BY ORDINAL_POSITION");
    for (auto& pr : pk.rows)
      for (size_t k = 0; k < t.cols.size(); ++k)
        if (t.cols[k].name == pr[0]) t.pk.push_back((int)k);
    out.push_back(std::move(t));
  }
  return out;
}

std::string createSql(const SrcTable& t, const std::string& target) {
  std::string s = "CREATE TABLE " + quoteId(target) + "." + quoteId(t.name) + " (";
  for (size_t i = 0; i < t.cols.size(); ++i) s += (i ? ", " : "") + quoteId(t.cols[i].name) + " " + targetType(t.cols[i]);
  if (!t.pk.empty()) {
    s += ", PRIMARY KEY (";
    for (size_t i = 0; i < t.pk.size(); ++i) s += (i ? ", " : "") + quoteId(t.cols[t.pk[i]].name);
    s += ")";
  }
  return s + ")";
}

// SELECT list for the snapshot: values in the text form warpstore loads.
std::string selectList(const SrcTable& t) {
  std::string s;
  for (size_t i = 0; i < t.cols.size(); ++i) {
    const SrcCol& c = t.cols[i];
    std::string e = quoteId(c.name);
    if (c.dataType == "bit") e = "CAST(" + e + " AS UNSIGNED)";
    else if (c.dataType == "datetime" || c.dataType == "timestamp") e = "DATE_FORMAT(" + e + ", '%Y-%m-%d %H:%i:%s.%f')";
    s += (i ? ", " : "") + e;
  }
  return s;
}

// ================================================================== binlog decoding

uint64_t le(const uint8_t* p, int n) {
  uint64_t v = 0;
  for (int k = 0; k < n; ++k) v |= (uint64_t)p[k] << (8 * k);
  return v;
}
uint64_t be(const uint8_t* p, int n) {
  uint64_t v = 0;
  for (int k = 0; k < n; ++k) v = (v << 8) | p[k];
  return v;
}

struct TableMap {
  std::string db, table;
  std::vector<uint8_t> types;
  std::vector<uint16_t> meta;
};

enum : uint8_t {
  T_DECIMAL = 0, T_TINY = 1, T_SHORT = 2, T_LONG = 3, T_FLOAT = 4, T_DOUBLE = 5, T_NULL = 6, T_TIMESTAMP = 7, T_LONGLONG = 8, T_INT24 = 9,
  T_DATE = 10, T_TIME = 11, T_DATETIME = 12, T_YEAR = 13, T_NEWDATE = 14, T_VARCHAR = 15, T_BIT = 16, T_TIMESTAMP2 = 17, T_DATETIME2 = 18,
  T_TIME2 = 19, T_TYPED_ARRAY = 20, T_VECTOR = 242, T_INVALID = 243, T_BOOL = 244, T_JSON = 245, T_NEWDECIMAL = 246, T_ENUM = 247, T_SET = 248,
  T_TINY_BLOB = 249, T_MEDIUM_BLOB = 250, T_LONG_BLOB = 251, T_BLOB = 252, T_VAR_STRING = 253, T_STRING = 254, T_GEOMETRY = 255
};

TableMap parseTableMap(const uint8_t* b, size_t n, uint64_t& tableId) {
  TableMap m;
  size_t i = 0;
  tableId = le(b, 6);
  i = 8;
  const int dl = b[i++];
  m.db.assign((const char*)b + i, dl);
  i += dl + 1;
  const int tl = b[i++];
  m.table.assign((const char*)b + i, tl);
  i += tl + 1;
  std::string s((const char*)b, n);
  const uint64_t nc = lenenc(s, i);
  m.types.assign(b + i, b + i + nc);
  i += nc;
  const uint64_t ml = lenenc(s, i);
  const size_t mend = i + ml;
  m.meta.assign(nc, 0);
  for (uint64_t c = 0; c < nc; ++c) {
    switch (m.types[c]) {
      case T_FLOAT: case T_DOUBLE: case T_BLOB: case T_GEOMETRY: case T_JSON: case T_TIMESTAMP2: case T_DATETIME2: case T_TIME2:
      case T_TINY_BLOB: case T_MEDIUM_BLOB: case T_LONG_BLOB: case T_VECTOR:
        m.meta[c] = b[i++];
        break;
      case T_VARCHAR: case T_VAR_STRING: case T_BIT:
        m.meta[c] = (uint16_t)le(b + i, 2);
        i += 2;
        break;
      case T_NEWDECIMAL: case T_STRING: case T_ENUM: case T_SET:
        m.meta[c] = (uint16_t)(b[i] << 8 | b[i + 1]);  // (precision, scale) / (real type, length)
        i += 2;
        break;
      default: break;
    }
  }
  (void)mend;
  return m;
}

// MySQL binary JSON -> JSON text.
void jsonValue(const uint8_t* base, size_t len, uint8_t type, size_t off, std::string& out, int depth = 0);
void jsonString(const std::string& s, std::string& out) {
  out += '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out += '\\', out += (char)c;
    else if (c < 0x20) {
      char b[8];
      snprintf(b, sizeof b, "\\u%04x", c);
      out += b;
    } else out += (char)c;
  }
  out += '"';
}
uint64_t varlen(const uint8_t* p, size_t& i) {
  uint64_t v = 0;
  for (int sh = 0; sh < 35; sh += 7) {
    const uint8_t c = p[i++];
    v |= (uint64_t)(c & 0x7f) << sh;
    if (!(c & 0x80)) break;
  }
  return v;
}
void jsonContainer(const uint8_t* base, size_t len, bool object, bool large, std::string& out, int depth) {
  const int w = large ? 4 : 2;
  if (len < (size_t)2 * w) throw MyErr("bad binary JSON");
  const uint32_t count = (uint32_t)le(base, w);
  size_t keyEntries = 2 * w, valEntries = keyEntries + (object ? (size_t)count * (w + 2) : 0);
  out += object ? '{' : '[';
  for (uint32_t k = 0; k < count; ++k) {
    if (k) out += ',';
    if (object) {
      const size_t ko = le(base + keyEntries + k * (w + 2), w), kl = le(base + keyEntries + k * (w + 2) + w, 2);
      jsonString(std::string((const char*)base + ko, kl), out);
      out += ':';
    }
    const uint8_t* ve = base + valEntries + (size_t)k * (1 + w);
    const uint8_t vt = ve[0];
    const bool inlined = vt == 0x04 || vt == 0x05 || vt == 0x06 || (large && (vt == 0x07 || vt == 0x08));
    if (inlined) jsonValue(ve + 1, w, vt, 0, out, depth + 1);
    else jsonValue(base, len, vt, le(ve + 1, w), out, depth + 1);
  }
  out += object ? '}' : ']';
}
void jsonValue(const uint8_t* base, size_t len, uint8_t type, size_t off, std::string& out, int depth) {
  if (depth > 100) throw MyErr("binary JSON nested too deeply");
  const uint8_t* p = base + off;
  char b[64];
  switch (type) {
    case 0x00: case 0x01: jsonContainer(p, len - off, true, type == 0x01, out, depth); break;
    case 0x02: case 0x03: jsonContainer(p, len - off, false, type == 0x03, out, depth); break;
    case 0x04: out += p[0] == 0 ? "null" : p[0] == 1 ? "true" : "false"; break;
    case 0x05: out += std::to_string((int16_t)le(p, 2)); break;
    case 0x06: out += std::to_string((uint16_t)le(p, 2)); break;
    case 0x07: out += std::to_string((int32_t)le(p, 4)); break;
    case 0x08: out += std::to_string((uint32_t)le(p, 4)); break;
    case 0x09: out += std::to_string((int64_t)le(p, 8)); break;
    case 0x0a: out += std::to_string((uint64_t)le(p, 8)); break;
    case 0x0b: {  // shortest round-trip form, ".0" kept on whole numbers (as MySQL prints them)
      double d;
      memcpy(&d, p, 8);
      auto r = std::to_chars(b, b + sizeof b, d);
      std::string v(b, r.ptr);
      if (v.find_first_of(".eEn") == std::string::npos) v += ".0";
      out += v;
      break;
    }
    case 0x0c: {
      size_t i = 0;
      const uint64_t l = varlen(p, i);
      jsonString(std::string((const char*)p + i, l), out);
      break;
    }
    case 0x0f: {  // opaque (dates, decimals ..): its text if printable, else base64-free hex
      size_t i = 1;
      const uint64_t l = varlen(p, i);
      std::string v((const char*)p + i, l);
      jsonString(v, out);
      break;
    }
    default: out += "null";
  }
}

std::string decimalText(const uint8_t* p, int prec, int scale, size_t& used) {
  static const int dig2bytes[10] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 4};
  const int intg = prec - scale, intg0 = intg / 9, frac0 = scale / 9, intg0x = intg - intg0 * 9, frac0x = scale - frac0 * 9;
  const size_t size = intg0 * 4 + dig2bytes[intg0x] + frac0 * 4 + dig2bytes[frac0x];
  std::vector<uint8_t> d(p, p + size);
  const bool neg = !(d[0] & 0x80);
  d[0] ^= 0x80;
  if (neg)
    for (auto& x : d) x ^= 0xff;
  size_t i = 0;
  std::string ip, fp;
  auto group = [&](int bytes, int digits) {
    const uint64_t v = be(&d[i], bytes);
    i += bytes;
    char b[16];
    snprintf(b, sizeof b, "%0*llu", digits, (unsigned long long)v);
    return std::string(b);
  };
  if (intg0x) ip += group(dig2bytes[intg0x], intg0x);
  for (int k = 0; k < intg0; ++k) ip += group(4, 9);
  for (int k = 0; k < frac0; ++k) fp += group(4, 9);
  if (frac0x) fp += group(dig2bytes[frac0x], frac0x);
  size_t z = 0;
  while (z + 1 < ip.size() && ip[z] == '0') ++z;
  ip = ip.empty() ? "0" : ip.substr(z);
  used = size;
  return (neg ? "-" : "") + ip + (scale ? "." + fp : "");
}

// One cell of a rows event -> SQL literal (or "NULL"). src: the column's schema (signedness, labels).
std::string cellSql(const uint8_t* b, size_t& i, uint8_t type, uint16_t meta, const SrcCol* src) {
  char buf[96];
  const bool uns = src && src->isUnsigned;
  switch (type) {
    case T_TINY: { const uint64_t v = b[i]; i += 1; return uns ? std::to_string(v) : std::to_string((int8_t)v); }
    case T_SHORT: { const uint64_t v = le(b + i, 2); i += 2; return uns ? std::to_string(v) : std::to_string((int16_t)v); }
    case T_INT24: {
      int32_t v = (int32_t)le(b + i, 3);
      i += 3;
      if (!uns && (v & 0x800000)) v |= ~0xffffff;
      return std::to_string(v);
    }
    case T_LONG: { const uint64_t v = le(b + i, 4); i += 4; return uns ? std::to_string(v) : std::to_string((int32_t)v); }
    case T_LONGLONG: { const uint64_t v = le(b + i, 8); i += 8; return uns ? std::to_string(v) : std::to_string((int64_t)v); }
    case T_FLOAT: { float f; memcpy(&f, b + i, 4); i += 4; snprintf(buf, sizeof buf, "%.9g", (double)f); return buf; }
    case T_DOUBLE: { double d; memcpy(&d, b + i, 8); i += 8; snprintf(buf, sizeof buf, "%.17g", d); return buf; }
    case T_YEAR: { const int v = b[i]; i += 1; return std::to_string(v ? 1900 + v : 0); }
    case T_NEWDECIMAL: {
      size_t used = 0;
      std::string s = decimalText(b + i, meta >> 8, meta & 0xff, used);
      i += used;
      return s;
    }
    case T_DATE: {
      const uint32_t v = (uint32_t)le(b + i, 3);
      i += 3;
      snprintf(buf, sizeof buf, "'%04u-%02u-%02u'", v >> 9, (v >> 5) & 15, v & 31);
      return buf;
    }
    case T_DATETIME2: {
      const int64_t v = (int64_t)be(b + i, 5) - 0x8000000000LL;
      i += 5;
      const int fb = (meta + 1) / 2;
      const uint64_t frac = fb ? be(b + i, fb) : 0;
      i += fb;
      const int64_t ymd = v >> 17, ym = ymd >> 5, hms = v % (1 << 17);
      int us = 0;
      if (meta) us = (int)(frac * (meta == 1 || meta == 2 ? 10000 : meta == 3 || meta == 4 ? 100 : 1));
      snprintf(buf, sizeof buf, "'%04lld-%02lld-%02lld %02lld:%02lld:%02lld.%06d'", (long long)(ym / 13), (long long)(ym % 13), (long long)(ymd % 32),
               (long long)(hms >> 12), (long long)((hms >> 6) % 64), (long long)(hms % 64), us);
      return buf;
    }
    case T_TIMESTAMP2: {
      const time_t sec = (time_t)be(b + i, 4);
      i += 4;
      const int fb = (meta + 1) / 2;
      const uint64_t frac = fb ? be(b + i, fb) : 0;
      i += fb;
      int us = meta ? (int)(frac * (meta == 1 || meta == 2 ? 10000 : meta == 3 || meta == 4 ? 100 : 1)) : 0;
      struct tm tm;
      gmtime_r(&sec, &tm);
      snprintf(buf, sizeof buf, "'%04d-%02d-%02d %02d:%02d:%02d.%06d'", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, us);
      return buf;
    }
    case T_TIME2: {
      const int64_t v = (int64_t)be(b + i, 3) - 0x800000;
      i += 3 + (meta + 1) / 2;
      const int64_t a = v < 0 ? -v : v;
      snprintf(buf, sizeof buf, "'%s%02lld:%02lld:%02lld'", v < 0 ? "-" : "", (long long)((a >> 12) % 1024), (long long)((a >> 6) % 64), (long long)(a % 64));
      return buf;
    }
    case T_TIMESTAMP: {  // (pre-5.6 formats)
      const time_t sec = (time_t)le(b + i, 4);
      i += 4;
      struct tm tm;
      gmtime_r(&sec, &tm);
      snprintf(buf, sizeof buf, "'%04d-%02d-%02d %02d:%02d:%02d'", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
      return buf;
    }
    case T_DATETIME: {
      const uint64_t v = le(b + i, 8);
      i += 8;
      snprintf(buf, sizeof buf, "'%04llu-%02llu-%02llu %02llu:%02llu:%02llu'", (unsigned long long)(v / 10000000000ULL), (unsigned long long)(v / 100000000 % 100),
               (unsigned long long)(v / 1000000 % 100), (unsigned long long)(v / 10000 % 100), (unsigned long long)(v / 100 % 100), (unsigned long long)(v % 100));
      return buf;
    }
    case T_BIT: {
      const int bytes = (meta >> 8) + ((meta & 0xff) + 7) / 8;
      const uint64_t v = be(b + i, bytes);
      i += bytes;
      return std::to_string(v);
    }
    case T_VARCHAR: case T_VAR_STRING: {
      const int lb = meta < 256 ? 1 : 2;
      const size_t l = le(b + i, lb);
      i += lb;
      std::string s((const char*)b + i, l);
      i += l;
      return sqlStr(s);
    }
    case T_STRING: {
      uint8_t real = (uint8_t)(meta >> 8);
      uint32_t len = meta & 0xff;
      if ((real & 0x30) != 0x30) {
        len |= ((real & 0x30) ^ 0x30) << 4;
        real |= 0x30;
      }
      if (real == T_ENUM) {
        const uint64_t idx = le(b + i, len);
        i += len;
        if (src && idx >= 1 && idx <= src->enumLabels.size()) return sqlStr(src->enumLabels[idx - 1]);
        return idx ? sqlStr(std::to_string(idx)) : "''";
      }
      if (real == T_SET) {
        const uint64_t bits = le(b + i, len);
        i += len;
        std::string s;
        for (size_t k = 0; src && k < src->enumLabels.size() && k < 64; ++k)
          if (bits >> k & 1) s += (s.empty() ? "" : ",") + src->enumLabels[k];
        return sqlStr(s);
      }
      const int lb = len < 256 ? 1 : 2;
      const size_t l = le(b + i, lb);
      i += lb;
      std::string s((const char*)b + i, l);
      i += l;
      return sqlStr(s);
    }
    case T_BLOB: case T_TINY_BLOB: case T_MEDIUM_BLOB: case T_LONG_BLOB: case T_GEOMETRY: case T_JSON: case T_VECTOR: {
      const size_t l = le(b + i, meta);
      i += meta;
      std::string s;
      if (type == T_JSON) {
        if (l) jsonValue(b + i + 1, l - 1, b[i], 0, s);
        else s = "null";
      } else {
        s.assign((const char*)b + i, l);
      }
      i += l;
      return sqlStr(s);
    }
    default: throw MyErr("unsupported column type " + std::to_string(type) + " in the binlog");
  }
}

// ================================================================== the replication loop

struct Url {
  std::string user, pw, host, db;
  int port = 3306;
  std::set<std::string> tables;
  bool tls = true;
  uint32_t serverId = 0;
};

std::string urlDecode(const std::string& s) {
  std::string o;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      o += (char)std::stoi(s.substr(i + 1, 2), nullptr, 16);
      i += 2;
    } else {
      o += s[i];
    }
  }
  return o;
}

Url parseUrl(const std::string& u) {
  Url r;
  std::string s = u;
  if (s.rfind("mysql://", 0) != 0) throw SqlError("MySQL source URL must look like mysql://user:password@host:port/database", 1105);
  s = s.substr(8);
  std::string opts;
  if (size_t q = s.find('?'); q != std::string::npos) opts = s.substr(q + 1), s = s.substr(0, q);
  if (size_t at = s.rfind('@'); at != std::string::npos) {
    std::string cred = s.substr(0, at);
    s = s.substr(at + 1);
    size_t c = cred.find(':');
    r.user = urlDecode(cred.substr(0, c));
    if (c != std::string::npos) r.pw = urlDecode(cred.substr(c + 1));
  }
  size_t slash = s.find('/');
  if (slash == std::string::npos || slash + 1 >= s.size()) throw SqlError("MySQL source URL needs a database: mysql://user:pw@host:port/database", 1105);
  r.db = urlDecode(s.substr(slash + 1));
  std::string hp = s.substr(0, slash);
  if (size_t c = hp.rfind(':'); c != std::string::npos) r.port = std::atoi(hp.substr(c + 1).c_str()), hp = hp.substr(0, c);
  r.host = hp;
  for (size_t a = 0; a < opts.size();) {
    size_t b = opts.find('&', a);
    std::string kv = opts.substr(a, b == std::string::npos ? std::string::npos : b - a);
    a = b == std::string::npos ? opts.size() : b + 1;
    size_t e = kv.find('=');
    std::string k = kv.substr(0, e), v = e == std::string::npos ? "" : urlDecode(kv.substr(e + 1));
    if (k == "tables")
      for (size_t x = 0; x <= v.size();) {
        size_t y = v.find(',', x);
        if (y == std::string::npos) y = v.size();
        if (y > x) r.tables.insert(v.substr(x, y - x));
        x = y + 1;
      }
    else if (k == "ssl" || k == "tls") r.tls = v != "0" && v != "false";
    else if (k == "server_id") r.serverId = (uint32_t)std::stoul(v);
  }
  if (!r.serverId) r.serverId = 0x47000000u + (std::random_device{}() & 0xffffff);
  return r;
}

class Cdc {
 public:
  Cdc(Engine& e, const std::string& url, const std::string& target, const std::string& pipe, const std::atomic<bool>& running,
      const std::function<void(const CdcStatus&)>& report)
      : e_(e), u_(parseUrl(url)), target_(lower(target.empty() ? parseUrl(url).db : target)), pipe_(pipe), running_(running), report_(report) {}

  void run(bool follow) {
    e_.execute("CREATE DATABASE IF NOT EXISTS " + quoteId(target_));
    e_.execute("CREATE TABLE IF NOT EXISTS " + quoteId(target_) + "._warpstore_cdc_position (pipeline VARCHAR(128) PRIMARY KEY, file VARCHAR(255), pos BIGINT, "
               "updated DATETIME)");
    bool first = true;
    while (running_) {
      try {
        std::string file;
        uint32_t pos = 0;
        if (!follow || !loadPosition(file, pos)) {
          snapshot(file, pos);
          if (!follow) return;
        } else if (first) {
          MyConn q = connect();
          tables_ = indexTables(loadSchema(q, u_.db, u_.tables));
        }
        first = false;
        stream(file, pos);
      } catch (SqlError& ex) {
        if (!follow) throw;
        st_.state = "Retrying";
        st_.error = ex.what();
        report_(st_);
        for (int i = 0; i < 30 && running_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
    st_.state = "Stopped";
    report_(st_);
  }

 private:
  Engine& e_;
  Url u_;
  std::string target_, pipe_;
  const std::atomic<bool>& running_;
  std::function<void(const CdcStatus&)> report_;
  CdcStatus st_;
  std::map<std::string, SrcTable> tables_;  // source table name -> schema

  MyConn connect() { return MyConn(u_.host, u_.port, u_.user, u_.pw, "", u_.tls); }
  static std::map<std::string, SrcTable> indexTables(std::vector<SrcTable> v) {
    std::map<std::string, SrcTable> m;
    for (auto& t : v) m[t.name] = std::move(t);
    return m;
  }
  std::string tq(const std::string& t) const { return quoteId(target_) + "." + quoteId(t); }

  bool loadPosition(std::string& file, uint32_t& pos) {
    ResultSet r = e_.execute("SELECT file, pos FROM " + quoteId(target_) + "._warpstore_cdc_position WHERE pipeline = " + sqlStr(pipe_));
    const auto rows = r.rows;
    if (rows.empty() || rows[0][0] == kNullCell) return false;
    file = rows[0][0];
    pos = (uint32_t)std::stoul(rows[0][1]);
    return true;
  }
  std::string positionSql(const std::string& file, uint64_t pos) const {
    return "REPLACE INTO " + quoteId(target_) + "._warpstore_cdc_position VALUES (" + sqlStr(pipe_) + ", " + sqlStr(file) + ", " + std::to_string(pos) + ", NOW())";
  }

  // Initial copy: the binlog position first, then a consistent snapshot of every table (changes in
  // between are applied again from the binlog: idempotent for tables with a primary key).
  void snapshot(std::string& file, uint32_t& pos) {
    st_.state = "Snapshotting";
    report_(st_);
    MyConn c = connect();
    c.exec("SET SESSION time_zone = '+00:00'");
    MyConn::Result bs;
    try {
      bs = c.query("SHOW BINARY LOG STATUS");
    } catch (SqlError&) {
      bs = c.query("SHOW MASTER STATUS");  // (MySQL before 8.2)
    }
    if (bs.rows.empty()) throw MyErr("the source has no binary log (log_bin = OFF)", 1105);
    file = bs.rows[0][0];
    pos = (uint32_t)std::stoul(bs.rows[0][1]);
    auto fmt = c.query("SELECT @@global.binlog_format, @@global.binlog_row_image");
    if (!fmt.rows.empty() && (lower(fmt.rows[0][0]) != "row" || lower(fmt.rows[0][1]) != "full"))
      throw MyErr("the source needs binlog_format = ROW and binlog_row_image = FULL (it has " + fmt.rows[0][0] + " / " + fmt.rows[0][1] + ")", 1105);
    c.exec("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ");
    c.exec("START TRANSACTION WITH CONSISTENT SNAPSHOT");
    tables_ = indexTables(loadSchema(c, u_.db, u_.tables));
    for (auto& [name, t] : tables_) copyTable(c, t);
    c.exec("COMMIT");
    e_.execute(positionSql(file, pos));
    st_.position = file + ":" + std::to_string(pos);
    report_(st_);
  }

  void copyTable(MyConn& c, const SrcTable& t) {
    e_.execute("DROP TABLE IF EXISTS " + tq(t.name));
    e_.execute(createSql(t, target_));
    const std::string dir = (e_.dataDir().empty() ? std::string("/tmp/warpstore-cdc") : e_.dataDir() + "/pipeline-loads");
    mkdir(dir.c_str(), 0755);
    std::string chunk;
    int64_t inChunk = 0, part = 0;
    auto flush = [&] {
      if (chunk.empty()) return;
      const std::string path = dir + "/mysql-" + pipe_ + "-" + t.name + "-" + std::to_string(part++) + "-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".tsv";
      FILE* f = fopen(path.c_str(), "wb");
      if (!f) throw SqlError("cannot stage " + path, 1105);
      fwrite(chunk.data(), 1, chunk.size(), f);
      fclose(f);
      std::string cols;
      for (auto& sc : t.cols) cols += (cols.empty() ? "" : ", ") + quoteId(sc.name);
      e_.execute("LOAD DATA INFILE " + sqlStr(path) + " INTO TABLE " + tq(t.name) + " FIELDS TERMINATED BY '\\t' ESCAPED BY '\\\\' (" + cols + ")");
      if (e_.dataDir().empty()) unlink(path.c_str());
      chunk.clear();
      inChunk = 0;
    };
    std::vector<std::string> names;
    c.queryStream("SELECT " + selectList(t) + " FROM " + quoteId(u_.db) + "." + quoteId(t.name), names,
                  [&](std::vector<std::string>& row, std::vector<bool>& nul) {
                    for (size_t k = 0; k < row.size(); ++k) {
                      if (k) chunk += '\t';
                      if (nul[k]) { chunk += "\\N"; continue; }
                      for (char ch : row[k]) {
                        if (ch == '\\') chunk += "\\\\";
                        else if (ch == '\t') chunk += "\\t";
                        else if (ch == '\n') chunk += "\\n";
                        else if (ch == '\r') chunk += "\\r";
                        else if (ch == '\0') chunk += "\\0";
                        else chunk += ch;
                      }
                    }
                    chunk += '\n';
                    ++st_.rows;
                    if (++inChunk >= 200000) flush();
                  });
    flush();
  }

  // Follows the binlog from file:pos; applies each source transaction as one warpstore transaction.
  void stream(std::string file, uint32_t pos) {
    MyConn c = connect();
    auto cs = c.query("SELECT @@global.binlog_checksum");
    const bool crc = !cs.rows.empty() && lower(cs.rows[0][0]) != "none";
    c.exec("SET @master_binlog_checksum = @@global.binlog_checksum");
    c.exec("SET @source_binlog_checksum = @@global.binlog_checksum");
    c.exec("SET @master_heartbeat_period = 1000000000");
    c.exec("SET @source_heartbeat_period = 1000000000");
    c.binlogDump(file, pos, u_.serverId);
    st_.state = "Streaming";
    st_.error.clear();
    report_(st_);
    std::map<uint64_t, TableMap> maps;
    std::vector<std::string> txn;  // statements of the open source transaction
    while (running_) {
      if (!c.readable(500)) {  // (heartbeats come every second; this notices STOP)
        if (pendingPos_ && Clock::now() - lastPosWrite_ > std::chrono::seconds(1)) flushPosition(file, pendingPos_);
        continue;
      }
      std::string ev = c.readEvent();
      if (ev.size() < 19) continue;
      const uint8_t* h = (const uint8_t*)ev.data();
      const uint8_t type = h[4];
      const uint32_t nextPos = (uint32_t)le(h + 13, 4);
      size_t bodyLen = ev.size() - 19 - (crc && type != 15 ? 4 : 0);
      const uint8_t* body = h + 19;
      switch (type) {
        case 4: {  // ROTATE: next file
          if (pendingPos_) flushPosition(file, pendingPos_);
          file.assign((const char*)body + 8, bodyLen - 8);
          pos = (uint32_t)le(body, 8);
          continue;
        }
        case 19: {  // TABLE_MAP
          uint64_t id;
          TableMap m = parseTableMap(body, bodyLen, id);
          maps[id] = std::move(m);
          break;
        }
        case 30: case 31: case 32: case 23: case 24: case 25: {  // WRITE / UPDATE / DELETE rows (v2, v1)
          rowsEvent(type, body, bodyLen, maps, txn);
          break;
        }
        case 2: {  // QUERY: BEGIN / DDL / COMMIT (non-transactional engines)
          const uint8_t dbLen = body[8];
          const uint16_t varLen = (uint16_t)le(body + 11, 2);
          const std::string db((const char*)body + 13 + varLen, dbLen);
          const std::string sql((const char*)body + 13 + varLen + dbLen + 1, bodyLen - (13 + varLen + dbLen + 1));
          const std::string up = upper(sql.substr(0, 16));
          if (up.rfind("BEGIN", 0) == 0) break;
          if (up.rfind("COMMIT", 0) == 0) {
            commit(txn, file, nextPos);
            break;
          }
          ddl(db, sql, file, nextPos);
          break;
        }
        case 16:  // XID: the transaction committed
          commit(txn, file, nextPos);
          break;
        default: break;  // FORMAT_DESCRIPTION, GTID, PREVIOUS_GTIDS, HEARTBEAT, ROWS_QUERY, ..
      }
      if (nextPos) pos = nextPos;
    }
  }

  using Clock = std::chrono::steady_clock;
  uint32_t pendingPos_ = 0;  // position of source transactions that changed nothing replicated (written lazily)
  Clock::time_point lastPosWrite_ = Clock::now();
  void flushPosition(const std::string& file, uint32_t pos) {
    e_.execute(positionSql(file, pos));
    pendingPos_ = 0;
    lastPosWrite_ = Clock::now();
  }
  void commit(std::vector<std::string>& txn, const std::string& file, uint32_t pos) {
    if (txn.empty()) {  // nothing for us: remember the position, write it at most once a second
      pendingPos_ = pos;
      if (Clock::now() - lastPosWrite_ > std::chrono::seconds(1)) flushPosition(file, pos);
      return;
    }
    pendingPos_ = 0;
    lastPosWrite_ = Clock::now();
    e_.execute("BEGIN");
    try {
      for (auto& s : txn) e_.execute(s);
      e_.execute(positionSql(file, pos));
      e_.execute("COMMIT");
    } catch (...) {
      try { e_.execute("ROLLBACK"); } catch (...) {}
      throw;
    }
    ++st_.txns;
    st_.position = file + ":" + std::to_string(pos);
    report_(st_);
    txn.clear();
  }

  // DDL on the source: tables are re-copied (schema and current rows) or dropped / emptied.
  void ddl(const std::string& db, const std::string& sql, const std::string& file, uint32_t pos) {
    std::string s = sql;
    const std::string up = upper(s);
    auto word = [&](size_t& i) {
      while (i < s.size() && isspace((unsigned char)s[i])) ++i;
      std::string w;
      if (i < s.size() && s[i] == '`') {
        for (++i; i < s.size() && s[i] != '`'; ++i) w += s[i];
        ++i;
      } else {
        while (i < s.size() && (isalnum((unsigned char)s[i]) || s[i] == '_' || s[i] == '$' || s[i] == '.')) w += s[i++];
      }
      return w;
    };
    // the table a statement names: after CREATE TABLE [IF NOT EXISTS] / ALTER TABLE / DROP TABLE [IF EXISTS] / TRUNCATE [TABLE] / RENAME TABLE
    auto tableAfter = [&](const std::string& kw) -> std::string {
      size_t i = up.find(kw);
      if (i == std::string::npos) return "";
      i += kw.size();
      std::string w = word(i);
      if (upper(w) == "IF") {
        word(i);
        w = word(i);
        if (upper(w) == "EXISTS") w = word(i);
      }
      if (size_t d = w.find('.'); d != std::string::npos) {
        if (lower(w.substr(0, d)) != lower(u_.db)) return "";
        w = w.substr(d + 1);
      } else if (lower(db) != lower(u_.db)) {
        return "";
      }
      return w;
    };
    auto wanted = [&](const std::string& t) { return !t.empty() && (u_.tables.empty() || u_.tables.count(t)); };
    std::string t;
    if (up.rfind("CREATE TABLE", 0) == 0 || up.rfind("ALTER TABLE", 0) == 0 || up.rfind("RENAME TABLE", 0) == 0 ||
        up.rfind("CREATE INDEX", 0) == 0 || up.rfind("DROP INDEX", 0) == 0) {
      // schema changed: re-read it and copy the table again (the stream continues where it is;
      // re-applied changes are idempotent for tables with a primary key)
      t = up.rfind("CREATE TABLE", 0) == 0 ? tableAfter("CREATE TABLE") : up.rfind("ALTER TABLE", 0) == 0 ? tableAfter("ALTER TABLE") : "";
      MyConn c = connect();
      c.exec("SET SESSION time_zone = '+00:00'");
      auto now = indexTables(loadSchema(c, u_.db, u_.tables));
      for (auto& [name, st] : now)
        if (name == t || !tables_.count(name)) copyTable(c, st);
      for (auto& [name, st] : tables_)
        if (!now.count(name)) e_.execute("DROP TABLE IF EXISTS " + tq(name));
      tables_ = std::move(now);
    } else if (up.rfind("DROP TABLE", 0) == 0) {
      size_t i = up.find("DROP TABLE") + 10;
      std::string w = word(i);
      if (upper(w) == "IF") { word(i); w = word(i); }
      for (;;) {  // DROP TABLE a, b
        std::string name = w.find('.') != std::string::npos ? w.substr(w.find('.') + 1) : w;
        if (wanted(name) && tables_.count(name)) {
          e_.execute("DROP TABLE IF EXISTS " + tq(name));
          tables_.erase(name);
        }
        while (i < s.size() && isspace((unsigned char)s[i])) ++i;
        if (i >= s.size() || s[i] != ',') break;
        ++i;
        w = word(i);
      }
    } else if (up.rfind("TRUNCATE", 0) == 0) {
      size_t i = 8;
      std::string w = word(i);
      if (upper(w) == "TABLE") w = word(i);
      if (size_t d = w.find('.'); d != std::string::npos) w = w.substr(d + 1);
      if (wanted(w) && tables_.count(w)) e_.execute("TRUNCATE TABLE " + tq(w));
    }
    e_.execute(positionSql(file, pos));
  }

  void rowsEvent(uint8_t type, const uint8_t* b, size_t n, std::map<uint64_t, TableMap>& maps, std::vector<std::string>& txn) {
    const uint64_t id = le(b, 6);
    size_t i = 8;
    if (type >= 30) i += le(b + i, 2);  // v2: extra data (length includes its 2 bytes)
    auto it = maps.find(id);
    if (it == maps.end()) return;
    const TableMap& m = it->second;
    if (lower(m.db) != lower(u_.db)) return;
    auto st = tables_.find(m.table);
    if (st == tables_.end()) return;  // not replicated
    const SrcTable& t = st->second;
    std::string s((const char*)b, n);
    const uint64_t nc = lenenc(s, i);
    const size_t bm = (nc + 7) / 8;
    std::vector<uint8_t> present(b + i, b + i + bm), present2;
    i += bm;
    const bool update = type == 31 || type == 24;
    if (update) {
      present2.assign(b + i, b + i + bm);
      i += bm;
    }
    auto readRow = [&](const std::vector<uint8_t>& pres, std::vector<std::string>& out) {
      int np = 0;
      for (uint64_t c = 0; c < nc; ++c) np += pres[c / 8] >> (c % 8) & 1;
      const uint8_t* nulls = b + i;
      i += (np + 7) / 8;
      out.assign(nc, "NULL");
      int k = 0;
      for (uint64_t c = 0; c < nc; ++c) {
        if (!(pres[c / 8] >> (c % 8) & 1)) continue;
        const bool isNull = nulls[k / 8] >> (k % 8) & 1;
        ++k;
        if (isNull) continue;
        out[c] = cellSql(b, i, m.types[c], m.meta[c], c < t.cols.size() ? &t.cols[c] : nullptr);
      }
    };
    std::string cols;
    for (size_t c = 0; c < t.cols.size(); ++c) cols += (c ? ", " : "") + quoteId(t.cols[c].name);
    auto values = [&](const std::vector<std::string>& r) {
      std::string v = "(";
      for (size_t c = 0; c < t.cols.size(); ++c) v += (c ? ", " : "") + (c < r.size() ? r[c] : std::string("NULL"));
      return v + ")";
    };
    auto keyCond = [&](const std::vector<std::string>& r) {  // the row's primary key (all columns without one)
      std::string w;
      if (!t.pk.empty())
        for (int c : t.pk) w += (w.empty() ? "" : " AND ") + quoteId(t.cols[c].name) + " = " + r[c];
      else
        for (size_t c = 0; c < t.cols.size(); ++c)
          w += (w.empty() ? "" : " AND ") + (r[c] == "NULL" ? quoteId(t.cols[c].name) + " IS NULL" : quoteId(t.cols[c].name) + " = " + r[c]);
      return w;
    };
    std::string inserts;
    auto flushInserts = [&] {
      if (inserts.empty()) return;
      txn.push_back(std::string(t.pk.empty() ? "INSERT" : "REPLACE") + " INTO " + tq(t.name) + " (" + cols + ") VALUES " + inserts);
      inserts.clear();
    };
    std::string delKeys;
    auto flushDeletes = [&] {
      if (delKeys.empty()) return;
      txn.push_back("DELETE FROM " + tq(t.name) + " WHERE " + quoteId(t.cols[t.pk[0]].name) + " IN (" + delKeys + ")");
      delKeys.clear();
    };
    std::vector<std::string> before, after;
    while (i < n) {
      if (type == 30 || type == 23) {  // insert (REPLACE: the snapshot may hold it already)
        flushDeletes();
        readRow(present, after);
        inserts += (inserts.empty() ? "" : ", ") + values(after);
        ++st_.rows;
        if (inserts.size() > (1 << 20)) flushInserts();
      } else if (type == 32 || type == 25) {  // delete (single-column keys: batched into IN lists)
        readRow(present, before);
        flushInserts();
        if (t.pk.size() == 1) {
          delKeys += (delKeys.empty() ? "" : ", ") + before[t.pk[0]];
          if (delKeys.size() > (1 << 20)) flushDeletes();
        } else {
          txn.push_back("DELETE FROM " + tq(t.name) + " WHERE " + keyCond(before) + (t.pk.empty() ? " LIMIT 1" : ""));
        }
        ++st_.rows;
      } else {  // update: before image, after image
        readRow(present, before);
        readRow(present2, after);
        bool keySame = !t.pk.empty();
        for (int c : t.pk) keySame = keySame && before[c] == after[c];
        flushDeletes();
        if (keySame) {  // same key: the new version replaces it (batched like inserts)
          inserts += (inserts.empty() ? "" : ", ") + values(after);
          if (inserts.size() > (1 << 20)) flushInserts();
        } else {
          flushInserts();
          txn.push_back("DELETE FROM " + tq(t.name) + " WHERE " + keyCond(before) + (t.pk.empty() ? " LIMIT 1" : ""));
          txn.push_back(std::string(t.pk.empty() ? "INSERT" : "REPLACE") + " INTO " + tq(t.name) + " (" + cols + ") VALUES " + values(after));
        }
        ++st_.rows;
      }
    }
    flushDeletes();
    flushInserts();
  }
};

}  // namespace

void mysqlCdcRun(Engine& e, const std::string& url, const std::string& target, const std::string& pipeName, const std::atomic<bool>& running,
                 bool follow, const std::function<void(const CdcStatus&)>& report) {
  Cdc c(e, url, target, pipeName, running, report);
  c.run(follow);
}

}  // namespace gdb
