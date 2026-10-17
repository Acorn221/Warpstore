// Clustering: one aggregator + leaf nodes. Leaves are ordinary gpu-db servers;
// the aggregator (--leaves http://leaf1:port,http://leaf2:port[|http://leaf2-replica:port], their
// HTTP API ports) keeps the catalog and talks to them over the HTTP Data API.
//
//   * DDL runs on the aggregator (its empty table is the catalog entry) and on every leaf.
//     REFERENCE tables are replicated: every write goes to the aggregator and to every leaf.
//   * Sharded tables: INSERT rows and LOAD DATA lines go to leaf hash(shard key) % nleaves;
//     UPDATE / DELETE are broadcast.
//   * SELECT: a query whose sharded tables are co-located (one sharded table, or several joined on
//     their shard keys; reference tables anywhere) is pushed down whole: aggregations as partial
//     aggregates combined on the aggregator's GPU, other queries with ORDER BY / LIMIT merged there.
//     Anything else fetches each sharded table's needed columns (its single-table WHERE conjuncts
//     pushed down) into a temp table that stands in for it, and runs the query on the aggregator.
//   * Results travel in a compact binary format (/api/v2/query/binary).
//   * ADD LEAF 'url' / REBALANCE PARTITIONS redistribute rows to the new hash layout; a leaf
//     written as "primary|replica" fails over to its replica (PROMOTE) when the primary is down.
//     --redundancy 2: no replica processes; each leaf's shard is mirrored on the next leaf.
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <regex>
#include <set>
#include <shared_mutex>
#include <thread>

#include "engine.h"
#include "sources.h"
#include "sql.h"
#include "storage.h"

namespace gdb {

thread_local bool tLocalOnly = false;  // run this statement on the aggregator only
thread_local bool tRequirePush = false;
ExprP cloneExpr(const ExprP& e);  // engine.cu
Stmt cloneStmt(const Stmt& s);  // engine.cu  // planning a reshuffled query: it must be pushed down

namespace {

bool isAggName(const std::string& n) { return n == "SUM" || n == "COUNT" || n == "AVG" || n == "MIN" || n == "MAX"; }
bool isHostAggName(const std::string& n) {
  return n == "GROUP_CONCAT" || n == "JSON_ARRAYAGG" || n == "JSON_AGG" || n == "JSON_OBJECTAGG" || n == "STRING_AGG" ||
         n == "STDDEV" || n == "STD" || n == "STDDEV_POP" || n == "STDDEV_SAMP" || n == "VARIANCE" || n == "VAR_POP" ||
         n == "VAR_SAMP" || n == "ANY_VALUE" || n == "MEDIAN" || n == "BIT_AND" || n == "BIT_OR" || n == "BIT_XOR";
}
std::string plainName(const std::string& n) { return n.find('.') == std::string::npos ? n : n.substr(n.find('.') + 1); }
std::string dbOfName(const std::string& n) { return n.find('.') == std::string::npos ? currentDb() : n.substr(0, n.find('.')); }

std::string jsonEsc(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') o += '\\', o += (char)c;
    else if (c == '\n') o += "\\n";
    else if (c == '\t') o += "\\t";
    else if (c == '\r') o += "\\r";
    else if (c < 0x20) {
      char b[8];
      snprintf(b, sizeof b, "\\u%04x", c);
      o += b;
    } else o += (char)c;
  }
  return o + "\"";
}

// ---- SQL rendering of parsed expressions (pushdown); "" when something can't be rendered.
std::string quoteId(const std::string& s) { return "`" + s + "`"; }
std::string sqlLit(const Value& v) {
  if (v.isNull()) return "NULL";
  if (v.kind == Value::INT) return std::to_string(v.i);
  if (v.kind == Value::DBL) {
    char b[40];
    snprintf(b, sizeof b, "%.17g", v.d);
    return b;
  }
  std::string s = v.toString(), o = "'";
  for (char c : s) {
    if (c == '\'' || c == '\\') o += '\\';
    o += c;
  }
  return o + "'";
}
std::string sqlOf(const ExprP& e) {
  if (!e) return "";
  auto all = [&](size_t from) {
    std::string r;
    for (size_t i = from; i < e->args.size(); ++i) {
      std::string a = sqlOf(e->args[i]);
      if (a.empty()) return std::string("\x01");
      r += (i > from ? ", " : "") + a;
    }
    return r;
  };
  switch (e->kind) {
    case Expr::COL: return e->table.empty() ? quoteId(e->name) : quoteId(e->table) + "." + quoteId(e->name);
    case Expr::LIT: return sqlLit(e->lit);
    case Expr::STAR: return e->table.empty() ? "*" : quoteId(e->table) + ".*";
    case Expr::INTERVAL: return "INTERVAL " + std::to_string(e->lit.i) + " " + e->op;
    case Expr::BIN: {
      std::string a = sqlOf(e->args[0]), b = sqlOf(e->args[1]);
      if (a.empty() || b.empty()) return "";
      return "(" + a + " " + e->op + " " + b + ")";
    }
    case Expr::UNARY: {
      std::string a = sqlOf(e->args[0]);
      if (a.empty()) return "";
      return e->op == "NOT" ? "(NOT " + a + ")" : "(" + e->op + a + ")";
    }
    case Expr::BETWEEN: {
      std::string a = sqlOf(e->args[0]), b = sqlOf(e->args[1]), c = sqlOf(e->args[2]);
      if (a.empty() || b.empty() || c.empty()) return "";
      return "(" + a + (e->neg ? " NOT" : "") + " BETWEEN " + b + " AND " + c + ")";
    }
    case Expr::INLIST: {
      if (e->sub) return "";
      std::string a = sqlOf(e->args[0]), r = all(1);
      if (a.empty() || r == "\x01") return "";
      return "(" + a + (e->neg ? " NOT" : "") + " IN (" + r + "))";
    }
    case Expr::LIKE: {
      std::string a = sqlOf(e->args[0]), b = sqlOf(e->args[1]);
      if (a.empty() || b.empty()) return "";
      return "(" + a + (e->neg ? " NOT" : "") + " LIKE " + b + ")";
    }
    case Expr::CASE: {
      std::string r = "CASE";
      for (size_t i = 0; i + 1 < e->args.size(); i += 2) {
        std::string w = sqlOf(e->args[i]), t = sqlOf(e->args[i + 1]);
        if (w.empty() || t.empty()) return "";
        r += " WHEN " + w + " THEN " + t;
      }
      if (e->args.size() % 2) {
        std::string el = sqlOf(e->args.back());
        if (el.empty()) return "";
        r += " ELSE " + el;
      }
      return r + " END";
    }
    case Expr::FUNC: {
      if (e->window || e->sub || e->op.empty() || e->op[0] == '#' || e->op == "EXISTS") return "";
      if (e->op == "ISNULL" || e->op == "ISNOTNULL") {
        std::string a = sqlOf(e->args[0]);
        return a.empty() ? "" : "(" + a + (e->op == "ISNULL" ? " IS NULL)" : " IS NOT NULL)");
      }
      if (e->op == "CAST") {
        std::string a = sqlOf(e->args[0]);
        return a.empty() || e->castType.empty() ? "" : "CAST(" + a + " AS " + e->castType + ")";
      }
      if (!e->winOrder.empty() || e->castType == "SEP" || e->op == "TIMESTAMPDIFF" || e->op == "TIMESTAMPADD") return "";
      std::string r = all(0);
      if (r == "\x01") return "";
      return e->op + "(" + (e->distinct ? "DISTINCT " : "") + r + ")";
    }
    default: return "";
  }
}

bool hasSubquery(const ExprP& e) {
  if (!e) return false;
  if (e->sub || e->kind == Expr::SUBQUERY) return true;
  for (auto& a : e->args)
    if (hasSubquery(a)) return true;
  return false;
}

// A leaf's result: column names / types and row-major cells (NULL flagged), or (columnar
// transfers) typed columns.
struct FetchedCol {
  int cls = 2;  // 0: ints, 1: doubles, 2: strings
  std::vector<uint8_t> nul;
  std::vector<int64_t> i;
  std::vector<double> d;
  std::vector<std::string> s;
};
struct Fetched {
  std::vector<std::string> names;
  std::vector<LType> types;
  std::vector<std::string> cells;
  std::vector<uint8_t> nul;
  size_t nrows = 0;
  std::vector<FetchedCol> cols;  // non-empty: columnar (cells / nul unused)
};

}  // namespace

std::string encodeColumnar(const ResultSet& r, const std::vector<size_t>* rows) {
  // "GDB2", u32 ncols, per column u8 type + u16 name length + name, u64 nrows, then per column:
  // nrows NULL flags, then the values: int64 for INT / BIGINT / DATE / DATETIME, float64 for DOUBLE /
  // DECIMAL, else u32 length + bytes each.
  const size_t nc = r.names.size(), total = r.typed ? r.vals.size() : r.rows.size(), nr = rows ? rows->size() : total;
  auto at = [&](size_t i) { return rows ? (*rows)[i] : i; };
  std::string out = "GDB2";
  auto put32 = [&](uint32_t v) { out.append((const char*)&v, 4); };
  put32((uint32_t)nc);
  std::vector<int> cls(nc);  // 0: int64, 1: float64, 2: bytes
  for (size_t c = 0; c < nc; ++c) {
    const LType t = c < r.types.size() ? r.types[c] : LType::VARCHAR;
    cls[c] = !r.typed ? 2
           : t == LType::INT || t == LType::BIGINT || t == LType::DATE || t == LType::DATETIME ? 0
           : t == LType::DOUBLE || t == LType::DECIMAL ? 1 : 2;
    // Text rows travel as strings: their type byte says VARCHAR unless the values are typed.
    out += (char)(uint8_t)(r.typed ? t : (t == LType::VARCHAR ? t : LType::VARCHAR));
    uint16_t n = (uint16_t)r.names[c].size();
    out.append((const char*)&n, 2);
    out += r.names[c];
  }
  uint64_t nr64 = nr;
  out.append((const char*)&nr64, 8);
  for (size_t c = 0; c < nc; ++c) {
    const size_t a = out.size();
    out.resize(a + nr);
    for (size_t i = 0; i < nr; ++i) out[a + i] = r.typed ? (char)r.vals[at(i)][c].isNull() : (char)(r.rows[at(i)][c] == kNullCell);
    if (cls[c] < 2) {
      const size_t a8 = out.size();
      out.resize(a8 + nr * 8);
      char* p8 = out.data() + a8;
      for (size_t i = 0; i < nr; ++i) {
        const Value& v = r.vals[at(i)][c];
        if (cls[c] == 0) { int64_t x = v.isNull() ? 0 : v.kind == Value::DBL ? (int64_t)v.d : v.i; memcpy(p8 + i * 8, &x, 8); }
        else { double x = v.isNull() ? 0 : v.num(); memcpy(p8 + i * 8, &x, 8); }
      }
    } else {
      for (size_t i = 0; i < nr; ++i) {
        const bool nul = r.typed ? r.vals[at(i)][c].isNull() : r.rows[at(i)][c] == kNullCell;
        const std::string s = nul ? std::string() : r.typed ? r.vals[at(i)][c].toString() : r.rows[at(i)][c];
        put32((uint32_t)s.size());
        out += s;
      }
    }
  }
  return out;
}

namespace {

Fetched parseColumnar(const std::string& b) {
  Fetched f;
  if (b.size() < 8 || b.compare(0, 4, "GDB2") != 0) throw SqlError("bad leaf response", 1105);
  size_t p = 4;
  auto need = [&](size_t n) {
    if (p + n > b.size()) throw SqlError("truncated leaf response", 1105);
  };
  uint32_t nc;
  need(4);
  memcpy(&nc, b.data() + p, 4);
  p += 4;
  for (uint32_t c = 0; c < nc; ++c) {
    need(3);
    f.types.push_back((LType)(uint8_t)b[p]);
    uint16_t nl;
    memcpy(&nl, b.data() + p + 1, 2);
    p += 3;
    need(nl);
    f.names.push_back(b.substr(p, nl));
    p += nl;
  }
  uint64_t nr;
  need(8);
  memcpy(&nr, b.data() + p, 8);
  p += 8;
  f.nrows = nr;
  f.cols.resize(nc);
  for (uint32_t c = 0; c < nc; ++c) {
    FetchedCol& col = f.cols[c];
    const LType t = f.types[c];
    col.cls = t == LType::INT || t == LType::BIGINT || t == LType::DATE || t == LType::DATETIME ? 0
            : t == LType::DOUBLE || t == LType::DECIMAL ? 1 : 2;
    need(nr);
    col.nul.assign(b.data() + p, b.data() + p + nr);
    p += nr;
    if (col.cls < 2) {
      need(nr * 8);
      if (col.cls == 0) col.i.resize(nr), memcpy(col.i.data(), b.data() + p, nr * 8);
      else col.d.resize(nr), memcpy(col.d.data(), b.data() + p, nr * 8);
      p += nr * 8;
    } else {
      col.s.resize(nr);
      for (uint64_t r = 0; r < nr; ++r) {
        uint32_t n;
        need(4);
        memcpy(&n, b.data() + p, 4);
        p += 4;
        need(n);
        col.s[r].assign(b.data() + p, n);
        p += n;
      }
    }
  }
  if (nc == 0) f.cols.clear();
  return f;
}

Fetched parseBinary(const std::string& b) {
  Fetched f;
  if (b.size() < 8 || b.compare(0, 4, "GDB1") != 0) throw SqlError("bad leaf response", 1105);
  size_t p = 4;
  auto need = [&](size_t n) {
    if (p + n > b.size()) throw SqlError("truncated leaf response", 1105);
  };
  uint32_t nc;
  need(4);
  memcpy(&nc, b.data() + p, 4);
  p += 4;
  for (uint32_t c = 0; c < nc; ++c) {
    need(3);
    f.types.push_back((LType)(uint8_t)b[p]);
    uint16_t nl;
    memcpy(&nl, b.data() + p + 1, 2);
    p += 3;
    need(nl);
    f.names.push_back(b.substr(p, nl));
    p += nl;
  }
  uint64_t nr;
  need(8);
  memcpy(&nr, b.data() + p, 8);
  p += 8;
  f.nrows = nr;
  f.cells.reserve(nr * nc);
  f.nul.reserve(nr * nc);
  for (uint64_t i = 0; i < nr * nc; ++i) {
    uint32_t n;
    need(4);
    memcpy(&n, b.data() + p, 4);
    p += 4;
    if (n == 0xFFFFFFFFu) {
      f.cells.emplace_back();
      f.nul.push_back(1);
      continue;
    }
    need(n);
    f.cells.emplace_back(b.data() + p, n);
    f.nul.push_back(0);
    p += n;
  }
  return f;
}

// Typed value of a fetched cell (for temp tables built from leaf results).
Value typedCell(const Fetched& f, size_t r, size_t c) {
  if (!f.cols.empty()) {
    const FetchedCol& col = f.cols[c];
    if (col.nul[r]) return Value::Null();
    switch (f.types[c]) {
      case LType::INT: case LType::BIGINT: return Value::I(col.i[r]);
      case LType::DATE: return Value::Dt(col.i[r]);
      case LType::DATETIME: return Value::Ts(col.i[r]);
      case LType::DOUBLE: case LType::DECIMAL: return Value::D(col.d[r]);
      default: return Value::S(col.s[r]);
    }
  }
  const size_t i = r * f.names.size() + c;
  if (f.nul[i]) return Value::Null();
  const std::string& s = f.cells[i];
  switch (f.types[c]) {
    case LType::INT: case LType::BIGINT: return Value::I(std::atoll(s.c_str()));
    case LType::DOUBLE: case LType::DECIMAL: return Value::D(std::strtod(s.c_str(), nullptr));
    default: return Value::S(s);
  }
}

// Appends a leaf result to a table: fetched column i fills table column need[i] (-1: dropped).
// Typed columns go straight into the physical layout (no per-cell parsing).
void appendFetched(Table& tt, const Fetched& p, const std::vector<int>& need) {
  const auto& cm = tt.cols;
  bool raw = !p.cols.empty() && tt.autoIncCol < 0;
  for (auto& c : cm) raw = raw && c.pt != PType::VEC;
  if (raw) {
    std::vector<int64_t> ints(cm.size());
    std::vector<double> dbls(cm.size());
    for (size_t r = 0; r < p.nrows; ++r) {
      for (size_t c = 0; c < cm.size(); ++c) ints[c] = cm[c].pt == PType::I64 ? kNullI64 : kNullI32, dbls[c] = std::nan("");
      for (size_t i = 0; i < need.size() && i < p.names.size(); ++i) {
        const FetchedCol& col = p.cols[i];
        const int tc = need[i];
        if (tc < 0) continue;
        if (col.nul[r]) {
          if (cm[tc].notNull) throw SqlError("Column '" + cm[tc].name + "' cannot be null", 1048);
          tt.hasNull[tc] = 1;
          continue;
        }
        switch (cm[tc].lt) {
          case LType::VARCHAR: ints[tc] = tt.dicts[tc]->getOrAdd(col.cls == 2 ? col.s[r] : typedCell(p, r, i).toString()); break;
          case LType::DOUBLE: dbls[tc] = col.cls == 1 ? col.d[r] : col.cls == 0 ? (double)col.i[r] : std::strtod(col.s[r].c_str(), nullptr); break;
          case LType::DECIMAL:
            ints[tc] = std::llround((col.cls == 1 ? col.d[r] : col.cls == 0 ? (double)col.i[r] : std::strtod(col.s[r].c_str(), nullptr)) * pow10i(cm[tc].scale));
            break;
          default:
            if (col.cls == 2) {  // text (e.g. a date): parse it the usual way
              ints[tc] = std::atoll(col.s[r].c_str());
              if (cm[tc].lt == LType::DATE || cm[tc].lt == LType::DATETIME) { raw = false; break; }
            } else {
              ints[tc] = col.cls == 0 ? col.i[r] : (int64_t)col.d[r];
            }
            break;
        }
        if (!raw) break;
      }
      if (!raw) {  // fall back for the rest
        for (size_t r2 = r; r2 < p.nrows; ++r2) {
          std::vector<Value> row(cm.size(), Value::Null());
          for (size_t i = 0; i < need.size() && i < p.names.size(); ++i)
            if (need[i] >= 0) row[need[i]] = typedCell(p, r2, i);
          tt.appendRow(row);
        }
        return;
      }
      tt.appendRaw(ints, dbls);
    }
    return;
  }
  std::vector<Value> row(cm.size(), Value::Null());
  for (size_t r = 0; r < p.nrows; ++r) {
    for (auto& v : row) v = Value::Null();
    for (size_t i = 0; i < need.size() && i < p.names.size(); ++i)
      if (need[i] >= 0) row[need[i]] = typedCell(p, r, i);
    tt.appendRow(row);
  }
}

std::mutex gLeavesMu;  // guards Engine::leaves_ (failover / ADD LEAF)

}  // namespace

void Engine::setLeaves(const std::vector<std::string>& leaves) {
  {
    std::lock_guard<std::mutex> g(gLeavesMu);
    leaves_ = leaves;
    for (auto& l : leaves_)
      while (!l.empty() && l.back() == '/') l.pop_back();
  }
  startHealthChecks();
}

// --redundancy 2 (SingleStore's redundancy level): leaf i's shard is mirrored on leaf (i + 1) % N, in
// databases named <db>__r<i>. Writes go to both copies; reads fail over to the mirror while leaf i is
// unreachable. The leaf helpers below rename the database through tLeafDbSuffix.
int g_redundancy = 1;
thread_local std::string tLeafDbSuffix;
std::mutex gFailedMu;
std::set<std::string> gFailed;  // leaves (primary URLs) served from their mirror until resynced
std::shared_mutex gResyncMu;    // writes hold it shared; a resync exclusively (no write is missed)
static std::string leafDb(const std::string& db) { return tLeafDbSuffix.empty() ? db : db + tLeafDbSuffix; }
// SQL for a mirror: `db`.`t` qualified names point into the suffixed database.
static std::string leafSql(const std::string& db, const std::string& sql) {
  if (tLeafDbSuffix.empty()) return sql;
  std::string out = sql;
  const std::string from = "`" + db + "`.", to = "`" + db + tLeafDbSuffix + "`.";
  for (size_t p = 0; (p = out.find(from, p)) != std::string::npos; p += to.size()) out.replace(p, from.size(), to);
  return out;
}
static std::string mirrorUrl(std::vector<std::string>& leaves, size_t i) {
  std::lock_guard<std::mutex> g(gLeavesMu);
  const std::string& spec = leaves[(i + 1) % leaves.size()];
  return spec.substr(0, spec.find('|'));
}
static bool connectionError(const std::exception& e) { return std::string(e.what()).find("failed:") != std::string::npos; }
// Runs fn against leaf i's mirror (database names suffixed).
template <class F>
static auto onMirror(std::vector<std::string>& leaves, size_t i, F&& fn) -> decltype(fn(std::string())) {
  struct Suffix {
    std::string saved;
    explicit Suffix(size_t i) : saved(tLeafDbSuffix) { tLeafDbSuffix = "__r" + std::to_string(i); }
    ~Suffix() { tLeafDbSuffix = saved; }
  } sfx(i);
  return fn(mirrorUrl(leaves, i));
}

// Calls fn(url) on leaf i; a leaf written "primary|replica" whose primary can't be reached fails
// over: the replica is promoted and replaces it. With --redundancy 2, writes (write = true) also go
// to the mirror, and reads go to the mirror while the primary is down.
template <class F>
static auto leafCall(std::vector<std::string>& leaves, size_t i, F&& fn, bool write = false) -> decltype(fn(std::string())) {
  std::string spec;
  {
    std::lock_guard<std::mutex> g(gLeavesMu);
    spec = leaves[i];
  }
  const std::string primary = spec.substr(0, spec.find('|'));
  if (g_redundancy >= 2 && leaves.size() >= 2 && spec.find('|') == std::string::npos) {
    // A leaf that failed stays on its mirror until it is resynced (when it comes back its copy is stale).
    auto isFailed = [&] { std::lock_guard<std::mutex> g(gFailedMu); return gFailed.count(primary) > 0; };
    auto markFailed = [&] { std::lock_guard<std::mutex> g(gFailedMu); gFailed.insert(primary); };
    std::shared_lock<std::shared_mutex> wl(gResyncMu, std::defer_lock);
    if (write) wl.lock();
    if (isFailed()) return onMirror(leaves, i, fn);
    if (write) {
      bool primaryOk = true;
      decltype(fn(std::string())) r{};
      try {
        r = fn(primary);
      } catch (const SqlError& e) {
        if (!connectionError(e)) throw;
        primaryOk = false;
        markFailed();
        fprintf(stderr, "[gpu-db] leaf %zu: %s unreachable, writing to its mirror only\n", i, primary.c_str());
      }
      if (!primaryOk) return onMirror(leaves, i, fn);
      {  // the mirror's leaf is failed: its copy is rebuilt when it is resynced
        std::lock_guard<std::mutex> g(gFailedMu);
        if (gFailed.count(mirrorUrl(leaves, i))) return r;
      }
      try {
        onMirror(leaves, i, fn);
      } catch (const SqlError& e) {  // the mirror's leaf is down: the write stands, without its copy
        if (!connectionError(e)) throw;
        fprintf(stderr, "[gpu-db] leaf %zu: mirror on %s unreachable, write not mirrored\n", i, mirrorUrl(leaves, i).c_str());
      }
      return r;
    }
    try {
      return fn(primary);
    } catch (const SqlError& e) {
      if (!connectionError(e)) throw;
      markFailed();
      fprintf(stderr, "[gpu-db] leaf %zu: %s unreachable, serving its shard from the mirror\n", i, primary.c_str());
      return onMirror(leaves, i, fn);
    }
  }
  try {
    return fn(primary);
  } catch (const SqlError& e) {
    const bool down = connectionError(e);  // connection error, not a SQL error
    if (!down || spec.find('|') == std::string::npos) throw;
    const std::string replica = spec.substr(spec.find('|') + 1);
    try {
      httpPost(replica + "/api/v2/exec", "{\"sql\":\"PROMOTE\"}");
    } catch (const SqlError& pe) {
      if (std::string(pe.what()).find("not a replica") == std::string::npos) throw;  // already promoted is fine
    }
    {
      std::lock_guard<std::mutex> g(gLeavesMu);
      if (leaves[i] == spec) leaves[i] = replica;
    }
    fprintf(stderr, "[gpu-db] leaf %zu: %s unreachable, failed over to %s\n", i, primary.c_str(), replica.c_str());
    return fn(replica);
  }
}

// Leaf health checks: every 2 s each leaf's primary is pinged; an unreachable leaf with a replica
// fails over to it (promoted) before a query has to find out.
void Engine::startHealthChecks() {
  std::thread([this] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      size_t n;
      {
        std::lock_guard<std::mutex> g(gLeavesMu);
        n = leaves_.size();
      }
      for (size_t i = 0; i < n; ++i) {
        if (g_redundancy >= 2) {  // a failed leaf that answers again: copy its data back, then it rejoins
          std::string primary;
          {
            std::lock_guard<std::mutex> g(gLeavesMu);
            primary = leaves_[i].substr(0, leaves_[i].find('|'));
          }
          bool failed;
          {
            std::lock_guard<std::mutex> g(gFailedMu);
            failed = gFailed.count(primary) > 0;
          }
          if (failed) {
            try {
              httpGet(primary + "/ping");
              resyncLeaf(i);
            } catch (const std::exception& e) {
            }
            continue;
          }
        }
        try {
          leafCall(leaves_, i, [&](const std::string& url) {
            httpGet(url + "/ping");
            return 0;
          });
        } catch (const std::exception& e) {
          fprintf(stderr, "[gpu-db] leaf %zu health check failed: %s\n", i, e.what());
        }
      }
    }
  }).detach();
}

static Fetched leafQueryUrl(const std::string& url, const std::string& db, const std::string& sql) {
  return parseBinary(httpPost(url + "/api/v2/query/binary", "{\"sql\":" + jsonEsc(leafSql(db, sql)) + ",\"database\":" + jsonEsc(leafDb(db)) + "}"));
}
// ---- Cluster data plane client: leaf traffic (columnar results, shuffles, bulk inserts) over a
// leaf's data port: pooled persistent TCP connections (one per thread and leaf), length-prefixed
// binary frames. Leaves without a data port (or unreachable ones) are called over HTTP.
struct LeafAddr {
  std::string host, user = "root", pw;
  int httpPort = 0;
};
static LeafAddr parseLeafUrl(const std::string& url) {
  LeafAddr a;
  std::string rest = url.substr(url.find("://") == std::string::npos ? 0 : url.find("://") + 3);
  rest = rest.substr(0, rest.find('/'));
  if (size_t at = rest.rfind('@'); at != std::string::npos) {
    const std::string cred = rest.substr(0, at);
    a.user = cred.substr(0, cred.find(':'));
    if (cred.find(':') != std::string::npos) a.pw = cred.substr(cred.find(':') + 1);
    rest = rest.substr(at + 1);
  }
  a.host = rest.substr(0, rest.rfind(':'));
  a.httpPort = rest.rfind(':') == std::string::npos ? 80 : std::atoi(rest.substr(rest.rfind(':') + 1).c_str());
  return a;
}
static int dataPortOf(const std::string& url) {
  static std::mutex mu;
  static std::map<std::string, std::pair<int, std::chrono::steady_clock::time_point>> ports;
  {
    std::lock_guard<std::mutex> g(mu);
    auto it = ports.find(url);
    if (it != ports.end() && (it->second.first > 0 || std::chrono::steady_clock::now() - it->second.second < std::chrono::seconds(30)))
      return it->second.first;
  }
  int port = 0;
  try {
    port = std::atoi(httpGet(url + "/api/v2/dataport").c_str());
  } catch (...) {
  }
  std::lock_guard<std::mutex> g(mu);
  ports[url] = {port, std::chrono::steady_clock::now()};
  return port;
}
static bool sendAll(int fd, const void* p, size_t n) {
  for (size_t o = 0; o < n;) {
    ssize_t w = ::send(fd, (const char*)p + o, n - o, MSG_NOSIGNAL);
    if (w <= 0) return false;
    o += (size_t)w;
  }
  return true;
}
static bool recvAll(int fd, void* p, size_t n) {
  for (size_t o = 0; o < n;) {
    ssize_t r = ::recv(fd, (char*)p + o, n - o, 0);
    if (r <= 0) return false;
    o += (size_t)r;
  }
  return true;
}
// One request/reply on fd; false if the connection broke.
static bool dataExchange(int fd, const std::string& path, const std::string& body, uint32_t& status, std::string& out) {
  uint32_t pl = (uint32_t)path.size();
  uint64_t bl = body.size(), ol;
  if (!sendAll(fd, &pl, 4) || !sendAll(fd, path.data(), pl) || !sendAll(fd, &bl, 8) || !sendAll(fd, body.data(), bl)) return false;
  if (!recvAll(fd, &status, 4) || !recvAll(fd, &ol, 8)) return false;
  out.resize(ol);
  return recvAll(fd, out.data(), ol);
}
// POST body to the leaf's API path (e.g. "/query/columnar"; over HTTP: /api/v2 + path).
static std::string dataCallRaw(const std::string& url, const std::string& path, const std::string& body);
static std::string dataCall(const std::string& url, const std::string& pathIn, const std::string& bodyIn) {
  if (tLeafDbSuffix.empty()) return dataCallRaw(url, pathIn, bodyIn);
  std::string path = pathIn, body = bodyIn;  // a mirror: database=<db> / "database":"<db>" get the suffix
  if (size_t p = path.find("database="); p != std::string::npos) {
    const size_t e = path.find('&', p);
    path.insert(e == std::string::npos ? path.size() : e, tLeafDbSuffix);
  }
  if (size_t p = body.find("\"database\":\""); p != std::string::npos && p < 4096) {
    const size_t e = body.find('"', p + 12);
    if (e != std::string::npos) body.insert(e, tLeafDbSuffix);
  }
  return dataCallRaw(url, path, body);
}
static std::string dataCallRaw(const std::string& url, const std::string& path, const std::string& body) {
  const int dport = dataPortOf(url);
  if (dport > 0) {
    struct Conns {  // url -> connected, authenticated socket; closed when the thread ends
      std::map<std::string, int> fds;
      ~Conns() {
        for (auto& [u, fd] : fds)
          if (fd > 0) close(fd);
      }
    };
    thread_local Conns conns;
    for (int attempt = 0; attempt < 2; ++attempt) {
      int& fd = conns.fds[url];
      if (fd <= 0) {
        const LeafAddr a = parseLeafUrl(url);
        addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(a.host.c_str(), std::to_string(dport).c_str(), &hints, &res) != 0 || !res) break;
        fd = socket(AF_INET, SOCK_STREAM, 0);
        const bool ok = fd >= 0 && connect(fd, res->ai_addr, res->ai_addrlen) == 0;
        freeaddrinfo(res);
        uint32_t st = 0;
        std::string resp;
        int one = 1;
        if (ok) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        if (!ok || !dataExchange(fd, "auth", a.user + ":" + a.pw, st, resp) || st != 200) {
          if (fd >= 0) close(fd);
          fd = 0;
          if (ok && st == 401) throw SqlError("data plane: access denied on " + url, 1045);
          break;  // HTTP below
        }
      }
      uint32_t status = 0;
      std::string out;
      if (!dataExchange(fd, path, body, status, out)) {  // stale connection: reconnect once
        close(fd);
        fd = 0;
        continue;
      }
      if (status != 200) {
        std::string msg = jsonField(out, "error");
        throw SqlError(msg.empty() ? "data plane status " + std::to_string(status) + " from " + url : msg, 1105);
      }
      return out;
    }
  }
  return httpPost(url + "/api/v2" + path, body, "application/octet-stream");
}

static Fetched leafQueryColsUrl(const std::string& url, const std::string& db, const std::string& sql) {
  // (dataCall suffixes the database for a mirror)
  return parseColumnar(dataCall(url, "/query/columnar", "{\"database\":" + jsonEsc(db) + ",\"sql\":" + jsonEsc(leafSql(db, sql)) + "}"));
}
static int64_t leafExecUrl(const std::string& url, const std::string& db, const std::string& sql) {
  if (!tLeafDbSuffix.empty()) {  // a mirror database is created on first use
    static std::mutex mu;
    static std::set<std::string> made;
    const std::string key = url + "/" + leafDb(db);
    bool fresh;
    {
      std::lock_guard<std::mutex> g(mu);
      fresh = made.insert(key).second;
    }
    if (fresh) httpPost(url + "/api/v2/exec", "{\"sql\":" + jsonEsc("CREATE DATABASE IF NOT EXISTS `" + leafDb(db) + "`") + "}");
  }
  std::string resp = httpPost(url + "/api/v2/exec", "{\"sql\":" + jsonEsc(leafSql(db, sql)) + ",\"database\":" + jsonEsc(leafDb(db)) + "}");
  return std::atoll(jsonField(resp, "rowsAffected").c_str());
}
// Runs fn(leaf index) for every leaf in parallel; rethrows the first error.
static void onLeaves(size_t n, const std::function<void(size_t)>& fn) {
  std::vector<std::thread> th;
  std::exception_ptr err;
  std::mutex mu;
  for (size_t i = 0; i < n; ++i)
    th.emplace_back([&, i] {
      try {
        fn(i);
      } catch (...) {
        std::lock_guard<std::mutex> g(mu);
        if (!err) err = std::current_exception();
      }
    });
  for (auto& t : th) t.join();
  if (err) std::rethrow_exception(err);
}

// Leaf of a row: hash of the shard key values (integers by value, everything else by text).
static uint64_t shardHashOf(const Table& t, const std::vector<std::string>& keyText) {
  uint64_t h = 0;
  for (size_t k = 0; k < t.shardKey.size(); ++k) {
    const ColumnMeta& c = t.cols[t.shardKey[k]];
    uint64_t x;
    if ((c.lt == LType::INT || c.lt == LType::BIGINT) && !keyText[k].empty()) x = shardHashInt(std::atoll(keyText[k].c_str()));
    else x = shardHashStr(keyText[k]);
    h = k ? shardCombine(h, x) : x;
  }
  return h;
}

bool Engine::isDistributed(const std::string& name) {
  if (leaves_.empty()) return false;
  auto t = cat_.find(name);
  if (!t) return false;
  const std::string db = lower(t->db);
  return !t->reference && db != "__tmp" && db != "information_schema";
}

static bool isReferenceTable(Catalog& cat, const std::string& name) {
  auto t = cat.find(name);
  return t && t->reference && lower(t->db) != "__tmp";
}

// The concatenation of the leaves' results, typed (for a temp table).
static ResultSet unionResults(const std::vector<Fetched>& parts) {
  ResultSet all;
  all.isResult = true;
  all.typed = true;
  all.names = parts[0].names;
  all.types = parts[0].types;
  for (auto& t : all.types)
    if (t == LType::DECIMAL) t = LType::DOUBLE;  // no scale in temp tables: keep the fraction
  for (auto& p : parts)
    for (size_t r = 0; r < p.nrows; ++r) {
      std::vector<Value> v;
      v.reserve(p.names.size());
      for (size_t c = 0; c < p.names.size(); ++c) v.push_back(typedCell(p, r, c));
      all.vals.push_back(std::move(v));
    }
  return all;
}

ResultSet Engine::clusterSelect(Stmt& s, const std::string& sql, bool explain) {
  resolveNaturalJoins(s);
  flattenDerived(s);
  // Base tables the statement reads (views expanded, CTE names excluded).
  std::vector<std::string> tables;
  std::set<std::string> colNames;
  bool star = false, anySub = false, viaView = false;
  std::function<void(const Stmt&, std::set<std::string>)> walk;
  std::function<void(const ExprP&, const std::set<std::string>&)> ex = [&](const ExprP& e, const std::set<std::string>& ctes) {
    if (!e) return;
    if (e->kind == Expr::COL) colNames.insert(lower(e->name));  // (COUNT(*)'s star reads no column)
    if (e->sub) {
      anySub = true;
      walk(*e->sub, ctes);
    }
    for (auto& a : e->args) ex(a, ctes);
    for (auto& a : e->partBy) ex(a, ctes);
    for (auto& [o, d] : e->winOrder) ex(o, ctes);
  };
  walk = [&](const Stmt& q, std::set<std::string> ctes) {
    for (auto& [n, c] : q.ctes) {
      ctes.insert(lower(n));
      walk(*c, ctes);
    }
    for (auto& f : q.from) {
      if (f.sub) {
        anySub = true;
        walk(*f.sub, ctes);
      } else if (!ctes.count(lower(f.name))) {
        std::string vsql;
        {
          std::lock_guard<std::mutex> g(cat_.mu);
          auto v = cat_.views.find(Catalog::key(f.name));
          if (v != cat_.views.end()) vsql = v->second;
        }
        if (!vsql.empty()) {
          viaView = true;
          Stmt vs = parseSql(vsql);
          walk(vs, ctes);
        } else {
          tables.push_back(f.name);
        }
      }
      ex(f.on, ctes);
    }
    for (auto& it : q.items) {
      if (it.e->kind == Expr::STAR) star = true;  // SELECT * / t.*
      ex(it.e, ctes);
    }
    ex(q.where, ctes);
    ex(q.having, ctes);
    for (auto& g : q.groupBy) ex(g, ctes);
    for (auto& o : q.orderBy) ex(o.e, ctes);
    for (auto& c : q.joinConds) ex(c, ctes);
    if (q.unionNext) walk(*q.unionNext, ctes);
  };
  walk(s, {});
  std::vector<std::string> dist;
  for (auto& t : tables)
    if (isDistributed(t) && std::find(dist.begin(), dist.end(), Catalog::key(t)) == dist.end()) dist.push_back(Catalog::key(t));
  auto runLocal = [&](const std::string& q) {
    tLocalOnly = true;
    try {
      ResultSet r = execute(q);
      tLocalOnly = false;
      return r;
    } catch (...) {
      tLocalOnly = false;
      throw;
    }
  };
  if (dist.empty()) return runLocal(sql);
  const size_t nl = leaves_.size();
  const std::string db = currentDb();

  // WHERE conjuncts of the top-level query that read only table `key` (rendered without qualifiers).
  auto singleTableWhere = [&](const std::string& key, const std::shared_ptr<Table>& t) {
    std::string tw;
    if (s.where && !s.unionNext) {
        std::string alias;
        int refs = 0;
        for (auto& f : s.from)
          if (!f.sub && Catalog::key(f.name) == key) {
            alias = f.alias.empty() ? plainName(f.name) : f.alias, ++refs;
            if (f.join == 1) refs += 100;  // NULL-extended side of a LEFT JOIN: no pushdown
          }
        std::vector<ExprP> conj;
        std::function<void(const ExprP&)> flat = [&](const ExprP& e) {
          if (e && e->kind == Expr::BIN && e->op == "AND") { flat(e->args[0]); flat(e->args[1]); }
          else if (e) conj.push_back(e);
        };
        flat(s.where);
        std::set<std::string> otherCols;  // an unqualified name another FROM table also has is ambiguous
        for (auto& f : s.from) {
          if (f.sub || Catalog::key(f.name) == key) continue;
          if (auto ot = cat_.find(f.name))
            for (auto& c : ot->cols) otherCols.insert(lower(c.name));
        }
        for (auto& e : conj) {
          if (refs != 1 || hasSubquery(e)) break;
          bool mine = true, any = false;
          std::function<void(const ExprP&)> chk = [&](const ExprP& x) {
            if (!x) return;
            if (x->kind == Expr::COL) {
              any = true;
              if (!x->table.empty()) mine &= strcasecmp(x->table.c_str(), alias.c_str()) == 0;
              else mine &= t->colIndex(x->name) >= 0 && !otherCols.count(lower(x->name));
            }
            if (x->kind == Expr::FUNC && isAggName(x->op)) mine = false;
            for (auto& a : x->args) chk(a);
          };
          chk(e);
          if (!mine || !any) continue;
          std::function<ExprP(const ExprP&)> strip = [&](const ExprP& x) -> ExprP {
            auto y = std::make_shared<Expr>(*x);
            if (y->kind == Expr::COL) y->table.clear();
            for (auto& a : y->args) a = strip(a);
            return y;
          };
          std::string w = sqlOf(strip(e));
          if (!w.empty()) tw += (tw.empty() ? "" : " AND ") + w;
        }
      }
    return tw;
  };
  // ---- Reshuffles: sharded tables joined on columns other than their shard keys. Step by step,
  // until every sharded table is co-located: a table that is alone in its group is repartitioned
  // leaf to leaf by its join column into a temporary sharded table; otherwise a whole co-located
  // group is joined on the leaves and its result repartitioned by the next join key (multi-stage).
  // The rewritten query is then pushed down.
  int shardedRefs = 0;
  for (auto& f : s.from) shardedRefs += !f.sub && isDistributed(f.name);
  if (!tRequirePush && shardedRefs >= 2 && !anySub && !viaView && s.ctes.empty() && !s.unionNext && !star) {
    bool simple = true;
    for (auto& f : s.from) simple &= !f.sub && (f.join == 0 || f.join == 1);  // inner and LEFT joins (self-joins too)
    if (simple) {
      static std::atomic<int> shufSeq{0};
      std::vector<std::string> created;  // temporary sharded tables (aggregator + leaves)
      const Stmt orig = cloneStmt(s);
      std::vector<std::string> steps;
      auto dropAll = [&] {
        for (auto& name : created) {
          try { runLocal("DROP TABLE IF EXISTS " + name); } catch (...) {}
          onLeaves(nl, [&](size_t i) {
            try { leafCall(leaves_, i, [&](const std::string& url) { return leafExecUrl(url, db, "DROP TABLE IF EXISTS " + name); }); } catch (...) {}
          });
        }
        s = cloneStmt(orig);
      };
      auto intLike = [](LType t) { return t == LType::INT || t == LType::BIGINT; };
      std::vector<std::string> urls;
      {
        std::lock_guard<std::mutex> g(gLeavesMu);
        for (auto& l : leaves_) urls.push_back(l.substr(0, l.find('|')));
      }
      std::string leafList;
      for (auto& u : urls) leafList += (leafList.empty() ? "" : ",") + jsonEsc(u);
      try {
        bool done = false;
        for (int iter = 0; iter < 12 && !done; ++iter) {
          // Resolve column references against the current FROM list.
          auto refOf = [&](const ExprP& c) -> std::pair<int, int> {
            if (!c || c->kind != Expr::COL) return {-1, -1};
            int hit = -1, col = -1;
            for (size_t i = 0; i < s.from.size(); ++i) {
              const std::string alias = s.from[i].alias.empty() ? plainName(s.from[i].name) : s.from[i].alias;
              if (!c->table.empty() && strcasecmp(c->table.c_str(), alias.c_str()) != 0) continue;
              auto t = cat_.find(s.from[i].name);
              int k = t ? t->colIndex(c->name) : -1;
              if (k < 0) continue;
              if (hit >= 0) return {-1, -1};
              hit = (int)i, col = k;
            }
            return {hit, col};
          };
          std::vector<int> fromDist;
          std::vector<std::shared_ptr<Table>> tabs(s.from.size());
          std::vector<int> shardCol(s.from.size(), -1);
          for (size_t i = 0; i < s.from.size(); ++i) {
            tabs[i] = cat_.find(s.from[i].name);
            if (isDistributed(s.from[i].name)) {
              fromDist.push_back((int)i);
              if (tabs[i]->shardKey.size() == 1) shardCol[i] = tabs[i]->shardKey[0];
            }
          }
          std::vector<ExprP> conj;
          std::function<void(const ExprP&)> flat = [&](const ExprP& e) {
            if (e && e->kind == Expr::BIN && e->op == "AND") { flat(e->args[0]); flat(e->args[1]); }
            else if (e) conj.push_back(e);
          };
          flat(s.where);
          for (auto& c : s.joinConds) flat(c);
          const size_t nWhere = conj.size();  // conjuncts after this come from LEFT JOIN ... ON
          for (auto& f : s.from) flat(f.on);
          std::vector<std::array<int, 4>> edges;  // sharded a.col = sharded b.col
          for (auto& e : conj) {
            if (e->kind != Expr::BIN || e->op != "=") continue;
            auto [t1, c1] = refOf(e->args[0]);
            auto [t2, c2] = refOf(e->args[1]);
            if (t1 < 0 || t2 < 0 || t1 == t2 || shardCol.size() <= (size_t)std::max(t1, t2)) continue;
            if (!isDistributed(s.from[t1].name) || !isDistributed(s.from[t2].name)) continue;
            edges.push_back({t1, c1, t2, c2});
          }
          std::vector<int> parent(s.from.size());
          for (size_t i = 0; i < parent.size(); ++i) parent[i] = (int)i;
          std::function<int(int)> root = [&](int x) { return parent[x] == x ? x : parent[x] = root(parent[x]); };
          for (auto& e : edges)
            if (shardCol[e[0]] == e[1] && shardCol[e[2]] == e[3]) parent[root(e[0])] = root(e[2]);
          std::map<int, std::vector<int>> groups;
          for (int i : fromDist) groups[root(i)].push_back(i);
          if (groups.size() <= 1) {
            done = true;
            break;
          }
          // The step: an edge between two groups.
          const std::array<int, 4>* pick = nullptr;
          int moveSide = -1;  // 0: move table / group of e[0] by e[1]; 1: of e[2] by e[3]; 2: both (lone tables)
          for (auto& e : edges) {
            if (root(e[0]) == root(e[2]) || intLike(tabs[e[0]]->cols[e[1]].lt) != intLike(tabs[e[2]]->cols[e[3]].lt)) continue;
            const bool aOn = shardCol[e[0]] == e[1], bOn = shardCol[e[2]] == e[3];
            if (aOn) moveSide = 1;
            else if (bOn) moveSide = 0;
            else moveSide = groups[root(e[0])].size() == 1 && groups[root(e[2])].size() == 1 ? 2 : 0;
            pick = &e;
            break;
          }
          // Repartition the members of group `g` joined together, keyed by (table fi, column col).
          // extraName / extraSql: a computed shard key column (kind 0: BIGINT, 1: DOUBLE, 2: string).
          auto repartition = [&](int fi, int col, const std::string& extraName = "", const std::string& extraSql = "", int extraKind = 0) {
            const bool extra = !extraName.empty();
            const std::vector<int> members = groups[root(fi)];
            if (members.size() > 1)
              for (int m : members)
                if (s.from[m].join != 0) throw SqlError("__nopush", 1105);  // a LEFT JOIN inside a materialized group
            const std::string shufName = "__shuf" + std::to_string(++shufSeq), tdb = lower(tabs[fi]->db);
            const std::string name = quoteId(tdb) + "." + quoteId(shufName);
            const bool single = members.size() == 1;
            // Columns of the members referenced anywhere in the query (plus the key).
            std::set<std::pair<int, int>> refs;
            refs.insert({fi, col});
            std::function<void(const ExprP&)> collect = [&](const ExprP& x) {
              if (!x) return;
              if (x->kind == Expr::COL) {
                auto r = refOf(x);
                if (r.first >= 0 && std::find(members.begin(), members.end(), r.first) != members.end()) refs.insert(r);
              }
              for (auto& a : x->args) collect(a);
              for (auto& a : x->partBy) collect(a);
            };
            for (auto& it : s.items) collect(it.e);
            collect(s.where);
            collect(s.having);
            for (auto& g : s.groupBy) collect(g);
            for (auto& o : s.orderBy) collect(o.e);
            for (auto& c : s.joinConds) collect(c);
            for (auto& f : s.from) collect(f.on);
            auto alias = [&](int m) { return s.from[m].alias.empty() ? plainName(s.from[m].name) : s.from[m].alias; };
            auto outName = [&](int m, int c) { return single ? tabs[m]->cols[c].name : alias(m) + "__" + tabs[m]->cols[c].name; };
            std::string ddl = "CREATE TABLE " + name + " (", sel;
            int key = 0, k = 0;
            for (auto [m, c] : refs) {
              if (extra && m == fi && c == col) continue;  // (col is a placeholder then)
              const ColumnMeta& cm = tabs[m]->cols[c];
              ddl += (k ? ", " : "") + quoteId(outName(m, c)) + " " + (cm.typeText.empty() ? std::string(ltypeName(cm.lt)) : cm.typeText);
              sel += (k ? ", " : "") + quoteId(alias(m)) + "." + quoteId(cm.name) + " AS " + quoteId(outName(m, c));
              if (m == fi && c == col) key = k;
              ++k;
            }
            if (extra) {  // a computed join key: the expression's value, as the shard key
              ddl += (k ? ", " : "") + quoteId(extraName) + (extraKind == 0 ? " BIGINT" : extraKind == 1 ? " DOUBLE" : " TEXT");
              sel += (k ? ", " : "") + extraSql + " AS " + quoteId(extraName);
              key = k++;
            }
            ddl += ", SHARD KEY (" + quoteId(extra ? extraName : outName(fi, col)) + "))";
            // The leaf query: the members joined (co-located), with the conjuncts that read only them.
            std::string from, where;
            for (int m : members)
              from += (from.empty() ? "" : ", ") + quoteId(lower(tabs[m]->db)) + "." + quoteId(lower(tabs[m]->name)) + " AS " + quoteId(alias(m));
            bool nullable = false;  // the NULL-extended side of a LEFT JOIN: WHERE must run after the join
            for (int m : members) nullable |= s.from[m].join == 1;
            for (size_t ci = 0; ci < nWhere && !nullable; ++ci) {
              const ExprP& c = conj[ci];
              bool mine = true;
              std::function<void(const ExprP&)> chk = [&](const ExprP& x) {
                if (!x) return;
                if (x->kind == Expr::COL) {
                  auto r = refOf(x);
                  mine &= r.first >= 0 && std::find(members.begin(), members.end(), r.first) != members.end();
                }
                if (x->kind == Expr::FUNC && isAggName(x->op)) mine = false;
                for (auto& a : x->args) chk(a);
              };
              chk(c);
              if (!mine || hasSubquery(c)) continue;
              std::string w = sqlOf(c);
              if (!w.empty()) where += (where.empty() ? "" : " AND ") + w;
            }
            runLocal(ddl);
            created.push_back(name);
            onLeaves(nl, [&](size_t i) { leafCall(leaves_, i, [&](const std::string& url) { return leafExecUrl(url, tdb, ddl); }); });
            const std::string body = "{\"sql\":" + jsonEsc("SELECT " + sel + " FROM " + from + (where.empty() ? "" : " WHERE " + where)) +
                                     ",\"database\":" + jsonEsc(tdb) + ",\"table\":" + jsonEsc(shufName) + ",\"key\":\"" + std::to_string(key) +
                                     "\",\"int\":\"" + ((extra ? extraKind == 0 : intLike(tabs[fi]->cols[col].lt)) ? "1" : "0") + "\",\"leaves\":[" + leafList + "]}";
            if (!explain) onLeaves(nl, [&](size_t i) { leafCall(leaves_, i, [&](const std::string& url) { return dataCall(url, "/reshuffle", body); }); });
            steps.push_back((single ? "reshuffle " + alias(fi) : "join {" + [&] { std::string a; for (int m : members) a += (a.empty() ? "" : ", ") + alias(m); return a; }() + "} on the leaves and reshuffle the result") +
                            " by " + (extra ? extraSql : tabs[fi]->cols[col].name));
            // Rewrite: the members become one FROM entry; their columns its columns.
            const std::string newAlias = single ? alias(fi) : "__j" + std::to_string(shufSeq.load());
            std::function<ExprP(const ExprP&)> rw = [&](const ExprP& x) -> ExprP {
              if (!x) return x;
              if (extra && x->kind == Expr::COL && x->name == extraName) return mkCol(newAlias, extraName);
              if (x->kind == Expr::COL) {
                auto r = refOf(x);
                if (r.first >= 0 && std::find(members.begin(), members.end(), r.first) != members.end())
                  return mkCol(newAlias, outName(r.first, r.second));
                return x;
              }
              auto y = std::make_shared<Expr>(*x);
              for (auto& a : y->args) a = rw(a);
              for (auto& a : y->partBy) a = rw(a);
              return y;
            };
            for (auto& it : s.items) {
              if (it.alias.empty()) it.alias = it.e->str();  // keep the output column name
              it.e = rw(it.e);
            }
            s.where = rw(s.where);
            s.having = rw(s.having);
            for (auto& g : s.groupBy) g = rw(g);
            for (auto& o : s.orderBy) o.e = rw(o.e);
            for (auto& c : s.joinConds) c = rw(c);
            for (auto& f : s.from) f.on = rw(f.on);
            std::vector<TableRef> nf;
            bool placed = false;
            for (size_t i = 0; i < s.from.size(); ++i) {
              if (std::find(members.begin(), members.end(), (int)i) == members.end()) {
                nf.push_back(s.from[i]);
                continue;
              }
              if (placed) continue;
              placed = true;
              nf.push_back(TableRef{tdb + "." + shufName, newAlias, nullptr, s.from[i].join, s.from[i].on});
            }
            s.from = nf;
          };
          // An equality between expressions of two sharded tables (a.x = b.y + 1000, a.d = ROUND(b.v, 2),
          // LOWER(a.s) = b.s, ..): a table is reshuffled with the expression's value as a computed shard
          // key column, so the equality becomes a column equality for the rounds below. Integer keys
          // hash as integers; other numbers are normalized to DOUBLE and strings hash as text, on both
          // sides (so both sides get computed keys).
          {
            static const std::set<std::string> strFns = {"CONCAT", "CONCAT_WS", "LOWER", "LCASE", "UPPER", "UCASE", "SUBSTRING", "SUBSTR",
                                                         "LEFT", "RIGHT", "TRIM", "LTRIM", "RTRIM", "REPLACE", "LPAD", "RPAD", "REVERSE",
                                                         "REGEXP_REPLACE", "SUBSTRING_INDEX", "MD5", "SHA1", "SHA2", "HEX"};
            // kind: 0 integer, 1 other number, 2 string, -1 unsupported; returns the one FROM entry read
            // (-1: none, -2: several / unsupported)
            std::function<int(const ExprP&, int&)> scan = [&](const ExprP& x, int& kind) -> int {
              if (!x || hasSubquery(x)) return kind = -1, -2;
              if (x->kind == Expr::COL) {
                auto [t, c] = refOf(x);
                if (t < 0) return -2;
                const LType lt = tabs[t]->cols[c].lt;
                kind = intLike(lt) ? 0 : lt == LType::DOUBLE || lt == LType::DECIMAL ? 1
                     : lt == LType::VARCHAR || lt == LType::DATE || lt == LType::DATETIME ? 2 : -1;  // dates: as text
                return t;
              }
              if (x->kind == Expr::LIT) {
                kind = x->lit.isNull() ? -1 : x->lit.kind == Value::INT ? 0 : x->lit.kind == Value::DBL ? 1 : x->lit.kind == Value::STR ? 2 : -1;
                return -1;
              }
              std::vector<int> kinds;
              int t = -1;
              for (auto& a : x->args) {
                int k = 0;
                const int ta = scan(a, k);
                if (ta == -2 || (ta >= 0 && t >= 0 && ta != t)) return -2;
                if (ta >= 0) t = ta;
                kinds.push_back(k);
              }
              const bool anyBad = std::find(kinds.begin(), kinds.end(), -1) != kinds.end();
              if (x->kind == Expr::BIN && (x->op == "+" || x->op == "-" || x->op == "*" || x->op == "/")) {
                if (anyBad || kinds[0] == 2 || kinds[1] == 2) return kind = -1, -2;
                kind = x->op != "/" && kinds[0] == 0 && kinds[1] == 0 ? 0 : 1;
              } else if (x->kind == Expr::UNARY && x->op == "-") {
                kind = kinds.empty() ? -1 : kinds[0];
              } else if (x->kind == Expr::FUNC && !isAggName(x->op) && !x->window) {
                kind = strFns.count(upper(x->op)) ? 2 : 1;  // other functions: numbers (normalized to DOUBLE)
              } else {
                return kind = -1, -2;
              }
              return t;
            };
            auto isKeyCol = [](const ExprP& x) { return x->kind == Expr::COL && x->name.rfind("__ek", 0) == 0; };
            bool changed = false;
            for (auto& e : conj) {
              if (changed) break;
              if (e->kind != Expr::BIN || e->op != "=") continue;
              int kx = 0, ky = 0;
              const int t0 = scan(e->args[0], kx), t1 = scan(e->args[1], ky);
              if (t0 < 0 || t1 < 0 || t0 == t1 || kx < 0 || ky < 0) continue;
              if (!isDistributed(s.from[t0].name) || !isDistributed(s.from[t1].name)) continue;
              // string = number compares as numbers (MySQL): both sides as DOUBLE then
              const int kind = kx == 0 && ky == 0 ? 0 : kx == 2 && ky == 2 ? 2 : 1;
              for (int side = 0; side < 2 && !changed; ++side) {
                const ExprP& x = e->args[side];
                const int tx = side == 0 ? t0 : t1;
                // integer keys: plain columns keep the column logic below; otherwise both sides get keys
                if (isKeyCol(x) || (kind == 0 && x->kind == Expr::COL)) continue;
                static std::atomic<int> ekSeq{0};
                const std::string name = "__ek" + std::to_string(++ekSeq);
                const std::string a = s.from[tx].alias.empty() ? plainName(s.from[tx].name) : s.from[tx].alias;
                const std::string keySql = kind == 1 ? "CAST(" + sqlOf(x) + " AS DOUBLE)" : sqlOf(x);
                if (keySql.empty() || keySql == "CAST( AS DOUBLE)") continue;
                // placeholder reference to the computed column; resolved against the shuffle table
                const Expr* target = x.get();
                std::function<void(ExprP&)> swapIn = [&](ExprP& z) {
                  if (!z) return;
                  if (z.get() == target) { z = mkCol(a, name); return; }
                  for (auto& w : z->args) swapIn(w);
                };
                swapIn(s.where);
                for (auto& c : s.joinConds) swapIn(c);
                for (auto& f : s.from) swapIn(f.on);
                repartition(tx, -1, name, keySql, kind);
                changed = true;
              }
            }
            if (changed) continue;  // FROM changed: recompute everything
          }
          if (!pick) break;  // groups not connected by a usable equality: the fetch path
          const auto e = *pick;
          if (moveSide == 2) {
            repartition(e[0], e[1]);
            continue;  // FROM indexes changed: recompute (the other side moves next round)
          }
          if (moveSide == 0) repartition(e[0], e[1]);
          else repartition(e[2], e[3]);
        }
        if (!done) throw SqlError("__nopush", 1105);
        tRequirePush = true;
        ResultSet r = clusterSelect(s, sql, explain);
        tRequirePush = false;
        if (explain)
          for (size_t k = steps.size(); k-- > 0;) r.rows.insert(r.rows.begin(), {"stage " + std::to_string(k + 1) + ": " + steps[k] + " (leaf to leaf)"});
        dropAll();
        return r;
      } catch (const SqlError& er) {
        tRequirePush = false;
        dropAll();
        if (std::string(er.what()) != "__nopush") throw;  // can't co-locate: the fetch path below
      } catch (...) {
        tRequirePush = false;
        dropAll();
        throw;
      }
    }
  }

  // ---- Co-location: the query (no subqueries / CTEs / views / unions) reads one sharded table, or
  // several joined pairwise on their single-column shard keys; reference tables are on every leaf.
  bool pushable = !anySub && !viaView && s.ctes.empty() && !s.unionNext;
  for (auto& f : s.from) pushable &= !f.sub;
  std::string fromSql, whereSql;
  if (pushable) {
    std::vector<int> fromDist;  // FROM indexes of sharded tables
    for (size_t i = 0; i < s.from.size(); ++i) {
      if (isDistributed(s.from[i].name)) fromDist.push_back((int)i);
      else if (!isReferenceTable(cat_, s.from[i].name)) pushable = false;  // e.g. a temporary table
    }
    if (pushable && fromDist.size() > 1) {
      // Union-find over sharded tables connected by shardKey = shardKey equalities.
      std::vector<int> parent(s.from.size());
      for (size_t i = 0; i < parent.size(); ++i) parent[i] = (int)i;
      std::function<int(int)> root = [&](int x) { return parent[x] == x ? x : parent[x] = root(parent[x]); };
      auto refOf = [&](const ExprP& c) -> std::pair<int, int> {  // (FROM index, column) of a column reference
        if (!c || c->kind != Expr::COL) return {-1, -1};
        int hit = -1, col = -1;
        for (size_t i = 0; i < s.from.size(); ++i) {
          const std::string alias = s.from[i].alias.empty() ? plainName(s.from[i].name) : s.from[i].alias;
          if (!c->table.empty() && strcasecmp(c->table.c_str(), alias.c_str()) != 0) continue;
          auto t = cat_.find(s.from[i].name);
          int k = t ? t->colIndex(c->name) : -1;
          if (k < 0) continue;
          if (hit >= 0) return {-1, -1};  // ambiguous
          hit = (int)i, col = k;
        }
        return {hit, col};
      };
      std::vector<ExprP> eqs;
      std::function<void(const ExprP&)> flat = [&](const ExprP& e) {
        if (e && e->kind == Expr::BIN && e->op == "AND") { flat(e->args[0]); flat(e->args[1]); }
        else if (e) eqs.push_back(e);
      };
      flat(s.where);
      for (auto& c : s.joinConds) flat(c);
      for (auto& f : s.from) flat(f.on);
      for (auto& e : eqs) {
        if (e->kind != Expr::BIN || e->op != "=") continue;
        auto [ta, ca] = refOf(e->args[0]);
        auto [tb, cb] = refOf(e->args[1]);
        if (ta < 0 || tb < 0 || ta == tb) continue;
        if (!isDistributed(s.from[ta].name) || !isDistributed(s.from[tb].name)) continue;
        auto A = cat_.find(s.from[ta].name), B = cat_.find(s.from[tb].name);
        if (A->shardKey.size() == 1 && B->shardKey.size() == 1 && A->shardKey[0] == ca && B->shardKey[0] == cb) parent[root(ta)] = root(tb);
      }
      for (int i : fromDist) pushable &= root(i) == root(fromDist[0]);
    }
    for (size_t i = 0; i < s.from.size() && pushable; ++i)  // RIGHT / FULL joins could lose rows across leaves
      if (s.from[i].join == 2 || s.from[i].join == 3) pushable = false;
    if (pushable) {
      for (size_t i = 0; i < s.from.size() && pushable; ++i) {
        const TableRef& f = s.from[i];
        std::string t = quoteId(lower(dbOfName(f.name))) + "." + quoteId(lower(plainName(f.name))) +
                        (f.alias.empty() ? "" : " AS " + quoteId(f.alias));
        if (i == 0) fromSql = t;
        else if (f.join == 0) fromSql += ", " + t;
        else {
          std::string on = sqlOf(f.on);
          if (on.empty()) pushable = false;
          fromSql += " LEFT JOIN " + t + " ON " + on;
        }
      }
      std::vector<std::string> w;
      if (s.where) w.push_back(sqlOf(s.where));
      for (auto& c : s.joinConds) w.push_back(sqlOf(c));
      for (auto& x : w) {
        if (x.empty()) pushable = false;
        whereSql += (whereSql.empty() ? "" : " AND ") + x;
      }
    }
  }
  bool hasWindow = false, hasDistinctAgg = false, plainAggOnly = true, hasAggs = false;
  std::function<void(const ExprP&)> scan = [&](const ExprP& e) {
    if (!e) return;
    if (e->kind == Expr::FUNC && e->window) hasWindow = true;
    if (e->kind == Expr::FUNC && isAggName(e->op)) {
      hasAggs = true;
      if (e->distinct) hasDistinctAgg = true;
      for (auto& a : e->args) scan(a);
      return;
    }
    if (e->kind == Expr::FUNC && isHostAggName(e->op)) plainAggOnly = false;
    for (auto& a : e->args) scan(a);
  };
  for (auto& it : s.items) scan(it.e);
  scan(s.having);
  for (auto& o : s.orderBy) scan(o.e);
  const std::string tail = std::string(" FROM ") + fromSql + (whereSql.empty() ? "" : " WHERE " + whereSql);
  auto explainRows = [&](std::vector<std::string> lines) {
    ResultSet r;
    r.isResult = true;
    r.names = {"plan"};
    r.types = {LType::VARCHAR};
    for (auto& l : lines) r.rows.push_back({l});
    return r;
  };
  auto runLeaves = [&](const std::string& leafSql) {
    std::vector<Fetched> parts(nl);
    onLeaves(nl, [&](size_t i) { parts[i] = leafCall(leaves_, i, [&](const std::string& url) { return leafQueryColsUrl(url, db, leafSql); }); });
    return parts;
  };
  auto runOuter = [&](Stmt& outer, const std::vector<Fetched>& parts) {
    ResultSet all = unionResults(parts);
    QueryCtx q;
    tLocalOnly = true;
    try {
      std::string tmp = makeTemp(all, q, "cl");
      outer.from = {TableRef{tmp, "__cl", nullptr, 0, nullptr}};
      ResultSet r = query(outer, q, false, false);
      tLocalOnly = false;
      dropTemps(q);
      return r;
    } catch (...) {
      tLocalOnly = false;
      dropTemps(q);
      throw;
    }
  };

  // ---- Pushed-down aggregation: partial aggregates per leaf, combined on the aggregator.
  if (pushable && (hasAggs || !s.groupBy.empty()) && !hasWindow && !hasDistinctAgg && plainAggOnly && !s.distinct) {
    auto resolveGroup = [&](const ExprP& g) -> ExprP {
      if (g->kind == Expr::LIT && g->lit.kind == Value::INT && g->lit.i >= 1 && g->lit.i <= (int64_t)s.items.size())
        return s.items[g->lit.i - 1].e;
      if (g->kind == Expr::COL && g->table.empty())
        for (auto& it : s.items)
          if (!it.alias.empty() && strcasecmp(it.alias.c_str(), g->name.c_str()) == 0) return it.e;
      return g;
    };
    std::vector<ExprP> groups;
    for (auto& g : s.groupBy) groups.push_back(resolveGroup(g));
    std::string sel, gb;
    bool ok = true;
    for (size_t i = 0; i < groups.size(); ++i) {
      std::string g = sqlOf(groups[i]);
      if (g.empty()) ok = false;
      sel += (sel.empty() ? "" : ", ") + g + " AS `__g" + std::to_string(i) + "`";
      gb += (gb.empty() ? "" : ", ") + g;
    }
    std::map<std::string, std::string> partOf;
    int np = 0;
    auto part = [&](const std::string& fn, const std::string& arg) {
      std::string k = fn + ":" + arg;
      auto it = partOf.find(k);
      if (it != partOf.end()) return it->second;
      std::string name = "__p" + std::to_string(np++);
      sel += (sel.empty() ? "" : ", ") + fn + "(" + arg + ") AS `" + name + "`";
      return partOf[k] = name;
    };
    std::function<ExprP(const ExprP&)> rw = [&](const ExprP& e) -> ExprP {
      if (!e) return e;
      for (size_t i = 0; i < groups.size(); ++i)
        if (e->str() == groups[i]->str()) return mkCol("", "__g" + std::to_string(i));
      if (e->kind == Expr::FUNC && isAggName(e->op)) {
        const std::string arg = e->args[0]->kind == Expr::STAR ? "*" : sqlOf(e->args[0]);
        if (arg.empty()) {
          ok = false;
          return e;
        }
        auto agg = [](const std::string& fn, ExprP a) {
          auto f = std::make_shared<Expr>();
          f->kind = Expr::FUNC;
          f->op = fn;
          f->args = {a};
          return f;
        };
        if (e->op == "COUNT") {
          auto c = std::make_shared<Expr>();
          c->kind = Expr::FUNC;
          c->op = "COALESCE";
          c->args = {agg("SUM", mkCol("", part("COUNT", arg))), mkLit(Value::I(0))};
          return c;
        }
        if (e->op == "SUM") return agg("SUM", mkCol("", part("SUM", arg)));
        if (e->op == "MIN") return agg("MIN", mkCol("", part("MIN", arg)));
        if (e->op == "MAX") return agg("MAX", mkCol("", part("MAX", arg)));
        return mkBin("/", agg("SUM", mkCol("", part("SUM", arg))), agg("SUM", mkCol("", part("COUNT", arg))));
      }
      if (e->kind == Expr::COL) ok = false;  // a column outside GROUP BY and aggregates
      auto y = std::make_shared<Expr>(*e);
      for (auto& a : y->args) a = rw(a);
      return y;
    };
    Stmt outer;
    outer.type = Stmt::SELECT;
    for (auto& it : s.items) outer.items.push_back({rw(it.e), it.alias.empty() ? it.e->str() : it.alias});
    outer.having = rw(s.having);
    for (auto& o : s.orderBy) {
      const bool alias = o.e->kind == Expr::COL && o.e->table.empty() &&
                         std::any_of(s.items.begin(), s.items.end(), [&](const SelectItem& it) { return !it.alias.empty() && strcasecmp(it.alias.c_str(), o.e->name.c_str()) == 0; });
      const bool pos = o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT;
      outer.orderBy.push_back({alias || pos ? o.e : rw(o.e), o.desc});
    }
    outer.limit = s.limit;
    outer.offset = s.offset;
    for (size_t i = 0; i < groups.size(); ++i) outer.groupBy.push_back(mkCol("", "__g" + std::to_string(i)));
    if (sel.empty()) sel = "COUNT(*) AS `__p0`";
    if (ok) {
      const std::string leafSql = "SELECT " + sel + tail + (gb.empty() ? "" : " GROUP BY " + gb);
      if (explain)
        return explainRows({"Distributed aggregation over " + std::to_string(nl) + " leaves (co-located; partial aggregates pushed down):",
                            "  leaf: " + leafSql, "  aggregator: combine the partials on the GPU, then HAVING / ORDER BY / LIMIT"});
      return runOuter(outer, runLeaves(leafSql));
    }
  }

  // ---- Pushed-down query: each leaf runs it (ORDER BY, LIMIT offset+limit); the aggregator merges,
  // re-sorts and applies LIMIT / OFFSET.
  if (pushable && !hasAggs && s.groupBy.empty() && !hasWindow && !star) {
    std::string sel;
    bool ok = true;
    for (size_t i = 0; i < s.items.size(); ++i) {
      std::string x = sqlOf(s.items[i].e);
      if (x.empty()) ok = false;
      sel += (i ? ", " : "") + x + " AS `__c" + std::to_string(i) + "`";
    }
    std::vector<std::pair<int, bool>> ord;  // ORDER BY: output columns, or hidden extra columns
    int ncols = (int)s.items.size();
    for (auto& o : s.orderBy) {
      int pos = -1;
      if (o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT) pos = (int)o.e->lit.i - 1;
      for (size_t i = 0; i < s.items.size() && pos < 0; ++i) {
        const SelectItem& it = s.items[i];
        if (o.e->kind == Expr::COL && o.e->table.empty() && !it.alias.empty() && strcasecmp(it.alias.c_str(), o.e->name.c_str()) == 0) pos = (int)i;
        else if (it.e->str() == o.e->str()) pos = (int)i;
      }
      if (pos < 0 && !s.distinct) {  // sort key not in the select list: carried as a hidden column
        std::string x = sqlOf(o.e);
        if (x.empty()) ok = false;
        sel += ", " + x + " AS `__c" + std::to_string(ncols) + "`";
        pos = ncols++;
      }
      if (pos < 0 || pos >= ncols) ok = false;
      ord.push_back({pos, o.desc});
    }
    if (ok) {
      std::string leafSql = std::string("SELECT ") + (s.distinct ? "DISTINCT " : "") + sel + tail;
      std::string ob;
      for (auto& [p, d] : ord) ob += (ob.empty() ? "" : ", ") + std::string("`__c") + std::to_string(p) + "`" + (d ? " DESC" : "");
      if (!ob.empty()) leafSql += " ORDER BY " + ob;
      if (s.limit >= 0) leafSql += " LIMIT " + std::to_string(s.limit + s.offset);
      if (explain)
        return explainRows({"Distributed query over " + std::to_string(nl) + " leaves (co-located, pushed down):", "  leaf: " + leafSql,
                            "  aggregator: merge" + std::string(ob.empty() ? "" : ", ORDER BY") + (s.limit >= 0 ? ", LIMIT" : "")});
      Stmt outer;
      outer.type = Stmt::SELECT;
      outer.distinct = s.distinct;
      for (size_t i = 0; i < s.items.size(); ++i)
        outer.items.push_back({mkCol("", "__c" + std::to_string(i)), s.items[i].alias.empty() ? s.items[i].e->str() : s.items[i].alias});
      for (auto& [p, d] : ord) outer.orderBy.push_back({mkCol("", "__c" + std::to_string(p)), d});
      outer.limit = s.limit;
      outer.offset = s.offset;
      return runOuter(outer, runLeaves(leafSql));
    }
  }

  // ---- Fetch path: each sharded table's needed columns into a temp table that replaces it.
  if (tRequirePush) throw SqlError("__nopush", 1105);
  const bool single = s.from.size() == 1 && !s.from[0].sub && s.joinConds.empty() && !s.unionNext && s.ctes.empty() &&
                      tables.size() == 1 && Catalog::key(s.from[0].name) == dist[0];
  const std::string where = single && s.where && !hasSubquery(s.where) ? sqlOf(s.where) : "";
  std::map<std::string, std::string> override;
  std::vector<std::string> temps;
  static std::atomic<int> seq{0};
  if (explain) {
    std::vector<std::string> lines;
    for (auto& k : dist) lines.push_back("fetch " + k + " from " + std::to_string(nl) + " leaves" + (single && !where.empty() ? " WHERE " + where : ""));
    lines.push_back("then run the query on the aggregator's GPU");
    return explainRows(lines);
  }
  auto cleanup = [&] {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& k : temps) cat_.tables.erase(k);
  };
  try {
    for (auto& key : dist) {
      auto t = cat_.get(key);
      std::vector<int> need;
      for (size_t c = 0; c < t->cols.size(); ++c)
        if (star || colNames.count(lower(t->cols[c].name))) need.push_back((int)c);
      if (need.empty()) need.push_back(0);
      std::string cols;
      for (int c : need) cols += (cols.empty() ? "" : ", ") + quoteId(t->cols[c].name);
      // Pushdown: the WHERE conjuncts of the top-level query that read only this table.
      std::string tw = single ? where : singleTableWhere(key, t);
      const std::string leafSql = "SELECT " + cols + " FROM " + quoteId(lower(t->db)) + "." + quoteId(lower(t->name)) +
                                  (tw.empty() ? "" : " WHERE " + tw);
      std::vector<Fetched> parts(nl);
      onLeaves(nl, [&](size_t i) {
        parts[i] = leafCall(leaves_, i, [&](const std::string& url) { return leafQueryColsUrl(url, lower(t->db), leafSql); });
      });
      std::vector<ColumnMeta> cm = t->cols;
      for (auto& c : cm) {
        c.notNull = false;
        c.autoInc = false;
        c.hasDefault = false;
      }
      const std::string name = "cl" + std::to_string(++seq);
      auto tt = std::make_shared<Table>(name, cm, -1, -1, std::vector<int>{}, std::vector<int>{}, 1, false);
      tt->db = "__tmp";
      for (auto& p : parts) appendFetched(*tt, p, need);
      tt->flush(true);
      {
        std::lock_guard<std::mutex> g(cat_.mu);
        cat_.tables["__tmp." + name] = tt;
      }
      temps.push_back("__tmp." + name);
      override[key] = "__tmp." + name;
    }
    tTableOverride = &override;
    tLocalOnly = true;
    ResultSet r = execute(sql);
    tTableOverride = nullptr;
    tLocalOnly = false;
    cleanup();
    return r;
  } catch (...) {
    tTableOverride = nullptr;
    tLocalOnly = false;
    cleanup();
    throw;
  }
}

// Inserts rows (SQL literal text per column) into a sharded table, routed by shard key.
int64_t Engine::clusterInsertRows(Table& t, const std::vector<std::string>& cols, const std::vector<std::vector<std::string>>& rows,
                                  const std::vector<std::vector<std::string>>& keyText, const std::string& verb,
                                  const std::string& tail) {
  const size_t nl = leaves_.size();
  static std::atomic<uint64_t> rr{0};
  std::vector<std::string> per(nl);
  for (size_t i = 0; i < rows.size(); ++i) {
    const size_t leaf = t.shardKey.empty() ? (size_t)(rr++ % nl) : (size_t)(shardHashOf(t, keyText[i]) % nl);
    std::string tuple = "(";
    for (size_t c = 0; c < rows[i].size(); ++c) tuple += (c ? "," : "") + rows[i][c];
    per[leaf] += (per[leaf].empty() ? "" : ",") + tuple + ")";
  }
  std::string colList;
  for (auto& c : cols) colList += (colList.empty() ? "" : ", ") + quoteId(c);
  const std::string head = verb + " INTO " + quoteId(lower(t.db)) + "." + quoteId(lower(t.name)) + " (" + colList + ") VALUES ";
  std::vector<int64_t> n(nl, 0);
  onLeaves(nl, [&](size_t i) {
    if (!per[i].empty())
      n[i] = leafCall(leaves_, i, [&](const std::string& url) { return leafExecUrl(url, lower(t.db), head + per[i] + tail); }, true);
  });
  int64_t total = 0;
  for (auto x : n) total += x;
  return total;
}

// Returns true when the statement was handled by the cluster layer.
bool Engine::clusterStatement(const std::string& sql, Stmt& s, ResultSet& out) {
  if (leaves_.empty() || tLocalOnly || replaying_) return false;
  const size_t nl = leaves_.size();
  auto local = [&](const std::string& q) {
    tLocalOnly = true;
    try {
      ResultSet r = execute(q);
      tLocalOnly = false;
      return r;
    } catch (...) {
      tLocalOnly = false;
      throw;
    }
  };
  const std::string db = currentDb();
  auto broadcast = [&](const std::string& q) {
    std::vector<int64_t> n(nl, 0);
    onLeaves(nl, [&](size_t i) { n[i] = leafCall(leaves_, i, [&](const std::string& url) { return leafExecUrl(url, db, q); }, true); });
    int64_t total = 0;
    for (auto x : n) total += x;
    return total;
  };
  auto selectPart = [&]() {  // the SELECT of an INSERT .. SELECT
    static const std::regex selRe(R"(\b(SELECT|WITH)\b)", std::regex::icase);
    std::smatch m;
    if (!std::regex_search(sql, m, selRe)) throw SqlError("INSERT .. SELECT: no SELECT found", 1064);
    return sql.substr(m.position(0));
  };
  switch (s.type) {
    case Stmt::SELECT:
      if (!s.outfile.empty()) return false;
      out = clusterSelect(s, sql, false);
      return true;
    case Stmt::EXPLAIN:
      if (!s.inner || s.inner->type != Stmt::SELECT) return false;
      out = clusterSelect(*s.inner, sql.substr(sql.find_first_not_of(" \t\r\n") + 7), true);
      return true;
    case Stmt::CREATE: {
      if (s.temporary) return false;  // temporary tables: aggregator only
      if (s.select) throw SqlError("CREATE TABLE .. AS SELECT in a cluster: create the table, then INSERT .. SELECT", 1235);
      out = local(sql);
      broadcast(sql);  // sharded tables are stored on the leaves; reference tables are replicated
      return true;
    }
    case Stmt::DROP: {
      bool any = false;
      for (auto& n : s.names) any |= isDistributed(n) || isReferenceTable(cat_, n);
      out = local(sql);
      if (any) broadcast(sql);
      return true;
    }
    case Stmt::ALTER: case Stmt::TRUNCATE: case Stmt::RENAME: {
      const std::string t = s.type == Stmt::RENAME ? (s.names.empty() ? "" : s.names[0]) : s.table;
      const bool d = isDistributed(t) || isReferenceTable(cat_, t);
      out = local(sql);
      if (d) broadcast(sql);
      return true;
    }
    case Stmt::CREATE_DB: case Stmt::DROP_DB:
      out = local(sql);
      broadcast(sql);
      return true;
    case Stmt::UPDATE: case Stmt::DELETE: {
      if (isReferenceTable(cat_, s.table)) {  // replicated: the aggregator's copy and every leaf
        out = local(sql);
        broadcast(sql);
        return true;
      }
      if (!isDistributed(s.table)) return false;
      auto t = cat_.get(s.table);
      for (auto& [col, e] : s.sets)
        for (int k : t->shardKey)
          if (strcasecmp(t->cols[k].name.c_str(), col.c_str()) == 0) throw SqlError("updating a shard key column is not supported", 1235);
      ResultSet r;
      r.affected = broadcast(sql);
      out = r;
      return true;
    }
    case Stmt::INSERT: {
      if (isReferenceTable(cat_, s.table)) {
        if (!s.select) {
          out = local(sql);
          broadcast(sql);
          return true;
        }
        ResultSet r = execute(selectPart());  // materialized once (it may read shards), inserted everywhere
        auto t = cat_.get(s.table);
        std::vector<std::string> cols = s.insertCols;
        if (cols.empty())
          for (auto& c : t->cols) cols.push_back(c.name);
        std::string vals;
        for (auto& row : r.rows) {
          std::string tuple = "(";
          for (size_t c = 0; c < row.size(); ++c) tuple += (c ? "," : "") + (row[c] == kNullCell ? std::string("NULL") : sqlLit(Value::S(row[c])));
          vals += (vals.empty() ? "" : ",") + tuple + ")";
        }
        if (vals.empty()) {
          out = ResultSet{};
          return true;
        }
        std::string colList;
        for (auto& c : cols) colList += (colList.empty() ? "" : ", ") + quoteId(c);
        const std::string ins = "INSERT INTO " + quoteId(lower(t->db)) + "." + quoteId(lower(t->name)) + " (" + colList + ") VALUES " + vals;
        out = local(ins);
        broadcast(ins);
        return true;
      }
      if (!isDistributed(s.table)) return false;
      auto t = cat_.get(s.table);
      std::vector<std::string> cols = s.insertCols;
      if (cols.empty())
        for (auto& c : t->cols) cols.push_back(c.name);
      std::vector<int> keyPos;
      for (int k : t->shardKey) {
        int pos = -1;
        for (size_t i = 0; i < cols.size(); ++i)
          if (strcasecmp(cols[i].c_str(), t->cols[k].name.c_str()) == 0) pos = (int)i;
        keyPos.push_back(pos);
      }
      std::vector<std::vector<std::string>> rows, keyText;
      if (s.select) {
        ResultSet r = execute(selectPart());  // through the cluster layer: it may read shards
        for (auto& row : r.rows) {
          std::vector<std::string> v, kt;
          for (size_t c = 0; c < row.size(); ++c) v.push_back(row[c] == kNullCell ? "NULL" : sqlLit(Value::S(row[c])));
          for (int p : keyPos) kt.push_back(p >= 0 && p < (int)row.size() && row[p] != kNullCell ? row[p] : "");
          rows.push_back(v);
          keyText.push_back(kt);
        }
      } else {
        for (auto& row : s.rows) {
          std::vector<std::string> v, kt;
          for (auto& e : row) {
            std::string x = sqlOf(e);
            if (x.empty()) throw SqlError("INSERT value can't be sent to the leaves: " + e->str(), 1235);
            v.push_back(x);
          }
          for (int p : keyPos) {
            if (p < 0 || p >= (int)row.size()) {
              kt.push_back("");
              continue;
            }
            const ExprP& e = row[p];
            if (e->kind == Expr::LIT) kt.push_back(e->lit.isNull() ? "" : e->lit.toString());
            else {
              ResultSet r = local("SELECT " + sqlOf(e));
              kt.push_back(r.rows.empty() || r.rows[0][0] == kNullCell ? "" : r.rows[0][0]);
            }
          }
          rows.push_back(v);
          keyText.push_back(kt);
        }
      }
      std::string tail;
      {
        static const std::regex dupRe(R"(\bON\s+DUPLICATE\s+KEY\s+UPDATE\b)", std::regex::icase);
        std::smatch m;
        if (!s.select && !s.onDup.empty() && std::regex_search(sql, m, dupRe)) tail = " " + sql.substr(m.position(0));
      }
      ResultSet r;
      r.affected = clusterInsertRows(*t, cols, rows, keyText, s.replaceInto ? "REPLACE" : s.insertIgnore ? "INSERT IGNORE" : "INSERT", tail);
      out = r;
      return true;
    }
    case Stmt::LOAD: {
      const bool ref = isReferenceTable(cat_, s.table);
      if ((!ref && !isDistributed(s.table)) || (s.setValue != "FS" && !s.setValue.empty()) || s.format == "JSON") return false;
      auto t = cat_.get(s.table);
      std::vector<int> keyField;
      for (int k : t->shardKey) {
        int f = k;
        if (!s.insertCols.empty()) {
          f = -1;
          for (size_t i = 0; i < s.insertCols.size(); ++i)
            if (strcasecmp(s.insertCols[i].c_str(), t->cols[k].name.c_str()) == 0) f = (int)i;
        }
        keyField.push_back(f);
      }
      std::ifstream in(s.path, std::ios::binary);
      if (!in) throw SqlError("File '" + s.path + "' not found", 29);
      const char delim = s.delim;
      std::string cols;
      for (auto& c : s.insertCols) cols += (cols.empty() ? "" : ", ") + c;
      std::string delimArg = delim == '\t' ? "\\t" : std::string(1, delim);
      auto esc = [](const std::string& x) {
        std::string o;
        static const char* hex = "0123456789ABCDEF";
        for (unsigned char c : x) {
          if (isalnum(c) || c == '-' || c == '_' || c == '.') o += (char)c;
          else o += '%', o += hex[c >> 4], o += hex[c & 15];
        }
        return o;
      };
      const std::string q = "/api/v2/load?database=" + esc(lower(t->db)) + "&table=" + esc(lower(t->name)) + "&delim=" + esc(delimArg) +
                            (cols.empty() ? "" : "&cols=" + esc(cols));
      std::vector<std::string> buf(nl);
      std::vector<int64_t> loaded(nl, 0);
      int64_t lineNo = 0;
      uint64_t rr = 0;
      auto flushAll = [&](bool force) {
        std::vector<size_t> todo;
        for (size_t i = 0; i < nl; ++i)
          if (!buf[i].empty() && (force || buf[i].size() > (64u << 20))) todo.push_back(i);
        if (todo.empty()) return;
        onLeaves(todo.size(), [&](size_t j) {
          const size_t i = todo[j];
          std::string resp = leafCall(leaves_, i, [&](const std::string& url) {
            std::string qq = q;  // (a mirror: the suffixed database, created on first use)
            if (!tLeafDbSuffix.empty())
              if (size_t p = qq.find("database="); p != std::string::npos) {
                const size_t e = qq.find('&', p);
                leafExecUrl(url, lower(t->db), "SELECT 1");
                qq.insert(e == std::string::npos ? qq.size() : e, tLeafDbSuffix);
              }
            return httpPost(url + qq, buf[i], "application/octet-stream");
          }, true);
          loaded[i] += std::atoll(jsonField(resp, "rowsAffected").c_str());
        });
        for (size_t i : todo) buf[i].clear();
      };
      std::string line;
      while (std::getline(in, line)) {
        if (lineNo++ < s.skipLines || line.empty()) continue;
        if (ref) {  // replicated: every leaf gets every line
          for (size_t i = 0; i < nl; ++i) buf[i] += line + '\n';
          if (buf[0].size() > (64u << 20)) flushAll(false);
          continue;
        }
        size_t leaf;
        if (t->shardKey.empty()) {
          leaf = rr++ % nl;
        } else {
          std::vector<std::string> kt;
          for (int f : keyField) {
            std::string v;
            int idx = 0;
            bool inQ = false;
            for (size_t p = 0; p <= line.size(); ++p) {
              if (p < line.size() && line[p] == '"') inQ = !inQ;
              if (p == line.size() || (line[p] == delim && !inQ)) {
                if (idx == f) break;
                ++idx;
                v.clear();
                continue;
              }
              v += line[p];
            }
            if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
            kt.push_back(v == "\\N" ? "" : v);
          }
          leaf = (size_t)(shardHashOf(*t, kt) % nl);
        }
        buf[leaf] += line;
        buf[leaf] += '\n';
        if (buf[leaf].size() > (64u << 20)) flushAll(false);
      }
      flushAll(true);
      ResultSet r;
      if (ref) {
        r = local(sql);  // the aggregator's copy
      } else {
        for (auto x : loaded) r.affected += x;
      }
      r.info = "Records: " + std::to_string(r.affected) + (ref ? " (replicated to every leaf)" : " across " + std::to_string(nl) + " leaves");
      out = r;
      return true;
    }
    default: return false;
  }
}

ResultSet Engine::clusterStatus() {
  ResultSet r;
  r.isResult = true;
  r.names = {"Leaf", "Endpoint", "Replica", "State"};
  r.types = {LType::BIGINT, LType::VARCHAR, LType::VARCHAR, LType::VARCHAR};
  std::vector<std::string> ls;
  {
    std::lock_guard<std::mutex> g(gLeavesMu);
    ls = leaves_;
  }
  for (size_t i = 0; i < ls.size(); ++i) {
    const std::string primary = ls[i].substr(0, ls[i].find('|'));
    std::string replica = ls[i].find('|') == std::string::npos ? "" : ls[i].substr(ls[i].find('|') + 1);
    if (replica.empty() && g_redundancy >= 2 && ls.size() >= 2)
      replica = "mirror on leaf " + std::to_string((i + 1) % ls.size()) + " (redundancy 2)";
    std::string state = "online";
    try {
      httpGet(primary + "/ping");
    } catch (const std::exception& e) {
      state = std::string("offline: ") + e.what();
    }
    r.rows.push_back({std::to_string(i), primary, replica.empty() ? kNullCell : replica, state});
  }
  return r;
}

static std::string encodeBinary(const Fetched& f, const std::vector<size_t>& rows);

// ADD LEAF 'url[|replica]' and REBALANCE PARTITIONS: every sharded table is redistributed to the
// current leaf count (rows whose leaf changes move); a new leaf gets the databases and tables first.
// Copies every table of the aggregator's catalog from (src, db + srcSfx) to (dst, db + dstSfx): the
// table is recreated there and its rows sent over the binary insert path.
void Engine::copyLeafTables(const std::string& src, const std::string& srcSfx, const std::string& dst, const std::string& dstSfx) {
  std::vector<std::shared_ptr<Table>> tabs;
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& [k, t] : cat_.tables)
      if (lower(t->db) != "__tmp") tabs.push_back(t);
  }
  std::set<std::string> made;
  for (auto& t : tabs) {
    const std::string db = lower(t->db), name = lower(t->name);
    const std::string sdb = db + srcSfx, ddb = db + dstSfx;
    if (made.insert(ddb).second) httpPost(dst + "/api/v2/exec", "{\"sql\":" + jsonEsc("CREATE DATABASE IF NOT EXISTS " + quoteId(ddb)) + "}");
    tLocalOnly = true;
    ResultSet c;
    try {
      c = execute("SHOW CREATE TABLE " + quoteId(db) + "." + quoteId(name));
    } catch (...) {
      tLocalOnly = false;
      throw;
    }
    tLocalOnly = false;
    if (c.rows.empty()) continue;
    leafExecUrl(dst, ddb, "DROP TABLE IF EXISTS " + quoteId(name));
    leafExecUrl(dst, ddb, c.rows[0][1]);
    Fetched f;
    try {
      f = leafQueryUrl(src, sdb, "SELECT * FROM " + quoteId(name));
    } catch (const SqlError&) {
      continue;  // not on the source (e.g. created while it was down)
    }
    if (f.nrows == 0) continue;
    std::vector<size_t> all(f.nrows);
    for (size_t r = 0; r < all.size(); ++r) all[r] = r;
    dataCall(dst, "/insert?database=" + ddb + "&table=" + name, encodeBinary(f, all));
  }
}

// --redundancy 2: leaf i is back. Its shard comes back from its mirror on leaf i + 1, the mirror it
// keeps for leaf i - 1 is rebuilt from that leaf, and it serves queries again.
void Engine::resyncLeaf(size_t i) {
  std::vector<std::string> ls;
  {
    std::lock_guard<std::mutex> g(gLeavesMu);
    ls = leaves_;
  }
  const size_t n = ls.size();
  auto url = [&](size_t k) { return ls[k].substr(0, ls[k].find('|')); };
  const size_t m = (i + 1) % n, p = (i + n - 1) % n;
  std::unique_lock<std::shared_mutex> wl(gResyncMu);  // writes wait until the leaf is back in sync
  fprintf(stderr, "[gpu-db] leaf %zu: back online, resyncing from its mirror on leaf %zu\n", i, m);
  copyLeafTables(url(m), "__r" + std::to_string(i), url(i), "");
  if (p != i) copyLeafTables(url(p), "", url(i), "__r" + std::to_string(p));
  {
    std::lock_guard<std::mutex> g(gFailedMu);
    gFailed.erase(url(i));
  }
  fprintf(stderr, "[gpu-db] leaf %zu: resynced\n", i);
}

// Rebuilds every leaf's mirror from its primary (after ADD LEAF / REBALANCE moved rows).
void Engine::rebuildMirrors() {
  std::vector<std::string> ls;
  {
    std::lock_guard<std::mutex> g(gLeavesMu);
    ls = leaves_;
  }
  const size_t n = ls.size();
  if (g_redundancy < 2 || n < 2) return;
  auto url = [&](size_t k) { return ls[k].substr(0, ls[k].find('|')); };
  onLeaves(n, [&](size_t p) { copyLeafTables(url(p), "", url((p + 1) % n), "__r" + std::to_string(p)); });
}

ResultSet Engine::clusterAdmin(const std::string& sql) {
  static const std::regex addRe(R"(^\s*(?:ALTER\s+CLUSTER\s+)?ADD\s+LEAF\s+'([^']+)'\s*;?\s*$)", std::regex::icase);
  static const std::regex rebRe(R"(^\s*REBALANCE(\s+ALL)?\s+PARTITIONS\b[\s\S]*$)", std::regex::icase);
  std::smatch m;
  std::string added;
  if (std::regex_match(sql, m, addRe)) {
    added = m[1].str();
    while (!added.empty() && added.back() == '/') added.pop_back();
  } else if (!std::regex_match(sql, rebRe)) {
    throw SqlError("expected ADD LEAF 'url' or REBALANCE PARTITIONS", 1064);
  }
  auto local = [&](const std::string& q) {
    tLocalOnly = true;
    try {
      ResultSet r = execute(q);
      tLocalOnly = false;
      return r;
    } catch (...) {
      tLocalOnly = false;
      throw;
    }
  };
  std::vector<std::shared_ptr<Table>> sharded, refs;
  std::set<std::string> dbs;
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    dbs = cat_.dbs;
    for (auto& [k, t] : cat_.tables) {
      if (lower(t->db) == "__tmp") continue;
      (t->reference ? refs : sharded).push_back(t);
    }
  }
  auto rowsAsValues = [&](const std::vector<std::string>& row) {
    std::string tuple = "(";
    for (size_t c = 0; c < row.size(); ++c) tuple += (c ? "," : "") + (row[c] == kNullCell ? std::string("NULL") : sqlLit(Value::S(row[c])));
    return tuple + ")";
  };
  if (!added.empty()) {
    const std::string url = added.substr(0, added.find('|'));
    for (auto& d : dbs) leafExecUrl(url, "gpudb", "CREATE DATABASE IF NOT EXISTS " + quoteId(d));
    for (auto* group : {&sharded, &refs})
      for (auto& t : *group) {
        ResultSet c = local("SHOW CREATE TABLE " + quoteId(lower(t->db)) + "." + quoteId(lower(t->name)));
        if (!c.rows.empty()) leafExecUrl(url, lower(t->db), c.rows[0][1]);
      }
    for (auto& t : refs) {  // reference tables: copy the aggregator's rows
      const std::string tq = quoteId(lower(t->db)) + "." + quoteId(lower(t->name));
      ResultSet r = local("SELECT * FROM " + tq);
      for (size_t b = 0; b < r.rows.size(); b += 5000) {
        std::string vals;
        for (size_t i = b; i < std::min(r.rows.size(), b + 5000); ++i) vals += (vals.empty() ? "" : ",") + rowsAsValues(r.rows[i]);
        leafExecUrl(url, lower(t->db), "INSERT INTO " + tq + " VALUES " + vals);
      }
    }
    std::lock_guard<std::mutex> g(gLeavesMu);
    leaves_.push_back(added);
  }
  // Redistribute: rows whose hash now maps to another leaf move there.
  const size_t nl = leaves_.size();
  int64_t moved = 0;
  for (auto& t : sharded) {
    if (t->shardKey.empty()) continue;  // keyless tables are round-robin anyway
    const std::string tdb = lower(t->db), tq = quoteId(tdb) + "." + quoteId(lower(t->name));
    for (size_t i = 0; i < nl; ++i) {
      Fetched f = leafCall(leaves_, i, [&](const std::string& url) { return leafQueryUrl(url, tdb, "SELECT * FROM " + tq); });
      if (f.nrows == 0) continue;
      const size_t nc = f.names.size();
      auto cell = [&](size_t r, size_t c) { return f.nul[r * nc + c] ? std::string() : f.cells[r * nc + c]; };
      auto lit = [&](size_t r, size_t c) { return f.nul[r * nc + c] ? std::string("NULL") : sqlLit(Value::S(f.cells[r * nc + c])); };
      auto destOf = [&](size_t r) {
        std::vector<std::string> kt;
        for (int k : t->shardKey) kt.push_back(cell(r, k));
        return (size_t)(shardHashOf(*t, kt) % nl);
      };
      std::vector<std::vector<size_t>> toLeaf(nl);  // rows per destination
      std::vector<std::string> movedKeys;             // primary keys of moving rows
      std::vector<size_t> keep;                       // rows that stay (tables without a primary key)
      for (size_t r = 0; r < f.nrows; ++r) {
        const size_t dest = destOf(r);
        if (dest == i) {
          keep.push_back(r);
          continue;
        }
        toLeaf[dest].push_back(r);
        if (t->pk >= 0) {  // integer keys as numbers (a quoted number doesn't compare with an integer column)
          const LType kt = t->cols[t->pk].lt;
          movedKeys.push_back((kt == LType::INT || kt == LType::BIGINT) && !f.nul[r * nc + t->pk] ? f.cells[r * nc + t->pk] : lit(r, t->pk));
        }
        ++moved;
      }
      if (keep.size() == f.nrows) continue;
      const std::string insQ = "/insert?database=" + tdb + "&table=" + lower(t->name);
      onLeaves(nl, [&](size_t d) {  // binary bulk inserts into the destinations
        if (toLeaf[d].empty()) return;
        leafCall(leaves_, d, [&](const std::string& url) { return dataCall(url, insQ, encodeBinary(f, toLeaf[d])); });
      });
      if (t->pk >= 0) {  // drop the moved rows from leaf i by primary key
        for (size_t b = 0; b < movedKeys.size(); b += 20000) {
          std::string in;
          for (size_t r = b; r < std::min(movedKeys.size(), b + 20000); ++r) in += (in.empty() ? "" : ",") + movedKeys[r];
          leafCall(leaves_, i, [&](const std::string& url) {
            return leafExecUrl(url, tdb, "DELETE FROM " + tq + " WHERE " + quoteId(t->cols[t->pk].name) + " IN (" + in + ")");
          });
        }
      } else {  // no primary key: rewrite leaf i with the rows that stay
        leafCall(leaves_, i, [&](const std::string& url) { return leafExecUrl(url, tdb, "TRUNCATE TABLE " + tq); });
        if (!keep.empty())
          leafCall(leaves_, i, [&](const std::string& url) { return dataCall(url, insQ, encodeBinary(f, keep)); });
      }
    }
  }
  // The binary inserts have no WAL records: checkpoint the leaves that persist (others ignore it).
  onLeaves(nl, [&](size_t i) {
    try {
      leafCall(leaves_, i, [&](const std::string& url) { return leafExecUrl(url, "gpudb", "SNAPSHOT"); }, true);
    } catch (...) {
    }
  });
  rebuildMirrors();  // --redundancy 2: the copies follow the new placement
  ResultSet r;
  r.affected = moved;
  r.info = std::to_string(moved) + " rows moved; " + std::to_string(nl) + " leaves" + (g_redundancy >= 2 ? ", mirrors rebuilt" : "");
  return r;
}

}  // namespace gdb

namespace gdb {

// ---- Leaf side of a shuffle: rows in the binary result format appended to a table (temporary
// shuffle tables and rebalancing; no WAL record: callers SNAPSHOT when they need durability).
int64_t Engine::insertBinary(const std::string& db, const std::string& table, const std::string& body) {
  if (body.compare(0, 4, "GDB2") == 0) {
    auto t0 = std::chrono::steady_clock::now();
    Fetched f = parseColumnar(body);
    auto t1 = std::chrono::steady_clock::now();
    auto t = cat_.get(db.empty() ? table : db + "." + table);
    std::unique_lock<FairSharedMutex> lk(t->mu);
    std::vector<int> need;
    for (auto& n : f.names) need.push_back(t->colIndex(n));
    appendFetched(*t, f, need);
    if (getenv("GPUDB_TRACE"))
      fprintf(stderr, "[trace] insert %zu rows (%zu bytes): parse %.1f ms, append %.1f ms\n", f.nrows, body.size(),
              std::chrono::duration<double, std::milli>(t1 - t0).count(),
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
    return (int64_t)f.nrows;
  }
  Fetched f = parseBinary(body);
  auto t = cat_.get(db.empty() ? table : db + "." + table);
  std::unique_lock<FairSharedMutex> lk(t->mu);
  std::vector<int> colOf;
  for (auto& n : f.names) colOf.push_back(t->colIndex(n));
  std::vector<Value> row(t->cols.size(), Value::Null());
  for (size_t r = 0; r < f.nrows; ++r) {
    for (auto& v : row) v = Value::Null();
    for (size_t c = 0; c < f.names.size(); ++c) {
      const size_t i = r * f.names.size() + c;
      if (colOf[c] >= 0 && !f.nul[i]) row[colOf[c]] = Value::S(f.cells[i]);
    }
    t->appendRow(row);
  }
  return (int64_t)f.nrows;
}

// Encodes rows [rows of a Fetched] back into the binary format.
static std::string encodeBinary(const Fetched& f, const std::vector<size_t>& rows) {
  std::string out = "GDB1";
  auto put32 = [&](uint32_t v) { out.append((const char*)&v, 4); };
  put32((uint32_t)f.names.size());
  for (size_t c = 0; c < f.names.size(); ++c) {
    out += (char)(uint8_t)f.types[c];
    uint16_t n = (uint16_t)f.names[c].size();
    out.append((const char*)&n, 2);
    out += f.names[c];
  }
  uint64_t nr = rows.size();
  out.append((const char*)&nr, 8);
  for (size_t r : rows)
    for (size_t c = 0; c < f.names.size(); ++c) {
      const size_t i = r * f.names.size() + c;
      if (f.nul[i]) {
        put32(0xFFFFFFFFu);
        continue;
      }
      put32((uint32_t)f.cells[i].size());
      out += f.cells[i];
    }
  return out;
}

// Leaf side of a shuffle: run `sql` locally, send each row to leaf hash(key column) % n's
// insert endpoint (into db.table). Request: {"sql","database","table","key","int","leaves":[..]}.
std::string Engine::reshuffle(const std::string& body) {
  const std::string sql = jsonField(body, "sql"), db = jsonField(body, "database"), table = jsonField(body, "table");
  const int key = std::atoi(jsonField(body, "key").c_str());
  const bool intKey = jsonField(body, "int") == "1";
  std::vector<std::string> leaves;
  for (size_t p = body.find("\"leaves\""); p != std::string::npos;) {  // ["url", ..]
    size_t a = body.find('[', p), b = body.find(']', a);
    std::string list = body.substr(a + 1, b - a - 1);
    for (size_t q = 0; (q = list.find('"', q)) != std::string::npos;) {
      size_t e = list.find('"', q + 1);
      leaves.push_back(list.substr(q + 1, e - q - 1));
      q = e + 1;
    }
    break;
  }
  if (leaves.empty()) throw SqlError("reshuffle: no target leaves", 1105);
  if (!db.empty()) execute("USE `" + db + "`");
  // One partition at a time: the leaf holds about 1/partitions of the shuffled rows at once, and
  // each partition's rows are on their way while the next one is scanned.
  auto esc = [](const std::string& x) {
    std::string o;
    static const char* hex = "0123456789ABCDEF";
    for (unsigned char c : x) {
      if (isalnum(c) || c == '-' || c == '_' || c == '.') o += (char)c;
      else o += '%', o += hex[c >> 4], o += hex[c & 15];
    }
    return o;
  };
  size_t total = 0;
  for (int part = 0; part < std::max(1, nparts_); ++part) {
    tTypedResult = true;
    tOnlyPart = nparts_ > 1 ? part : -1;
    ResultSet r;
    try {
      r = execute(sql);
    } catch (...) {
      tTypedResult = false;
      tOnlyPart = -1;
      throw;
    }
    tTypedResult = false;
    tOnlyPart = -1;
    const size_t nr = r.typed ? r.vals.size() : r.rows.size();
    total += nr;
    std::vector<std::vector<size_t>> per(leaves.size());
    for (size_t i = 0; i < nr; ++i) {
      uint64_t h;
      if (r.typed) {
        const Value& v = r.vals[i][key];
        h = v.isNull() ? shardHashStr("") : intKey ? shardHashInt(v.kind == Value::DBL ? (int64_t)v.d : v.i)
                                                   : shardHashStr(v.kind == Value::STR ? v.s : v.toString());
      } else {
        const std::string& k = r.rows[i][key] == kNullCell ? std::string() : r.rows[i][key];
        h = intKey && !k.empty() ? shardHashInt(std::atoll(k.c_str())) : shardHashStr(k);
      }
      per[h % leaves.size()].push_back(i);
    }
    onLeaves(leaves.size(), [&](size_t i) {
      constexpr size_t kChunk = 262144;  // a bounded encoded payload in flight per destination
      for (size_t b = 0; b < per[i].size(); b += kChunk) {
        std::vector<size_t> chunk(per[i].begin() + b, per[i].begin() + std::min(per[i].size(), b + kChunk));
        dataCall(leaves[i], "/insert?database=" + esc(db) + "&table=" + esc(table), encodeColumnar(r, &chunk));
      }
    });
  }
  const size_t nr = total;
  return "{\"rowsAffected\":" + std::to_string(nr) + "}";
}

}  // namespace gdb
