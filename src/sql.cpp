#include <charconv>
#include "sql.h"

#include "monitor.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <set>

namespace gdb {
std::string& currentDb();  // storage.cu: the connection's current database

// ---------------------------------------------------------------- utilities

std::string upper(std::string s) {
  for (auto& c : s) c = (char)toupper((unsigned char)c);
  return s;
}

bool parseDate(const std::string& s, int32_t& out) {
  int y, m, d;
  if (sscanf(s.c_str(), "%d-%d-%d", &y, &m, &d) != 3 || m < 1 || m > 12 || d < 1 || d > 31) return false;
  out = (int32_t)daysFromCivil(y, (unsigned)m, (unsigned)d);
  return true;
}

std::string formatDate(int64_t days) {
  int64_t y;
  unsigned m, d;
  civilFromDays(days, y, m, d);
  char buf[32];
  snprintf(buf, sizeof buf, "%04lld-%02u-%02u", (long long)y, m, d);
  return buf;
}

bool parseDateTime(const std::string& s, int64_t& us) {
  int y, mo, d, h = 0, mi = 0;
  double sec = 0;
  int n = sscanf(s.c_str(), "%d-%d-%d%*[ T]%d:%d:%lf", &y, &mo, &d, &h, &mi, &sec);
  if (n < 3 || mo < 1 || mo > 12 || d < 1 || d > 31) return false;
  if (n < 6) sec = 0;
  if (n < 5) mi = 0;
  if (n < 4) h = 0;
  us = daysFromCivil(y, (unsigned)mo, (unsigned)d) * kUsPerDay + (int64_t)h * 3600000000ll + (int64_t)mi * 60000000ll +
       (int64_t)std::llround(sec * 1e6);
  return true;
}

std::string formatDateTime(int64_t us) {
  int64_t days = floorDiv(us, kUsPerDay), rem = us - days * kUsPerDay;
  char buf[48];
  int n = snprintf(buf, sizeof buf, "%s %02lld:%02lld:%02lld", formatDate(days).c_str(), (long long)(rem / 3600000000ll),
                   (long long)(rem / 60000000ll % 60), (long long)(rem / 1000000ll % 60));
  if (rem % 1000000) snprintf(buf + n, sizeof buf - n, ".%06lld", (long long)(rem % 1000000));
  return buf;
}

static std::string fmtDouble(double v) {
  if (std::isnan(v)) return "NaN";
  char buf[64];
  if (std::fabs(v) < 1e15 && v == std::floor(v)) {
    snprintf(buf, sizeof buf, "%.1f", v);
    std::string s = buf;
    return s.substr(0, s.size() - 2);
  }
  // Shortest text that reads back as the same double (MySQL's DOUBLE output), exponent as 1e20 / 1e-7.
  auto r = std::to_chars(buf, buf + sizeof buf, v);
  std::string s(buf, r.ptr);
  if (auto e = s.find("e+"); e != std::string::npos) s.erase(e + 1, 1);
  return s;
}

std::string Value::toString() const {
  switch (kind) {
    case INT: return std::to_string(i);
    case DBL:
      if (scale >= 0 && !std::isnan(d)) {
        char b[64];
        snprintf(b, sizeof b, "%.*f", (int)scale, d);
        return b;
      }
      return fmtDouble(d);
    case STR: return s;
    case DATE: return formatDate(i);
    case DATETIME: return formatDateTime(i);
  }
  return "";
}

bool likeMatch(const std::string& s, const std::string& p) {
  // Iterative wildcard match with backtracking on the last '%'.
  size_t si = 0, pi = 0, star = std::string::npos, mark = 0;
  while (si < s.size()) {
    if (pi < p.size() && (p[pi] == '_' || p[pi] == s[si])) {
      ++si, ++pi;
    } else if (pi < p.size() && p[pi] == '%') {
      star = pi++, mark = si;
    } else if (star != std::string::npos) {
      pi = star + 1, si = ++mark;
    } else {
      return false;
    }
  }
  while (pi < p.size() && p[pi] == '%') ++pi;
  return pi == p.size();
}

ExprP mkCol(const std::string& table, const std::string& name) {
  auto e = std::make_shared<Expr>();
  e->kind = Expr::COL;
  e->table = table;
  e->name = name;
  return e;
}
ExprP mkLit(Value v) {
  auto e = std::make_shared<Expr>();
  e->kind = Expr::LIT;
  e->lit = std::move(v);
  return e;
}
ExprP mkBin(const std::string& op, ExprP a, ExprP b) {
  auto e = std::make_shared<Expr>();
  e->kind = Expr::BIN;
  e->op = op;
  e->args = {std::move(a), std::move(b)};
  return e;
}

std::string Expr::str() const {
  std::string r;
  switch (kind) {
    case COL: return table.empty() ? name : table + "." + name;
    case LIT: return lit.kind == Value::STR ? "'" + lit.s + "'" : lit.toString();
    case STAR: return "*";
    case INTERVAL: return "INTERVAL " + std::to_string(lit.i) + " " + op;
    case BIN: return "(" + args[0]->str() + " " + op + " " + args[1]->str() + ")";
    case UNARY: return op + "(" + args[0]->str() + ")";
    case FUNC:
      r = op + "(" + (distinct ? "DISTINCT " : "");
      for (size_t i = 0; i < args.size(); ++i) r += (i ? "," : "") + args[i]->str();
      return r + ")";
    case BETWEEN:
      return "(" + args[0]->str() + (neg ? " NOT" : "") + " BETWEEN " + args[1]->str() + " AND " + args[2]->str() + ")";
    case INLIST:
      r = "(" + args[0]->str() + (neg ? " NOT" : "") + " IN (";
      for (size_t i = 1; i < args.size(); ++i) r += (i > 1 ? "," : "") + args[i]->str();
      return r + "))";
    case LIKE: return "(" + args[0]->str() + (neg ? " NOT" : "") + " LIKE " + args[1]->str() + ")";
    case CASE:
      r = "CASE";
      for (size_t i = 0; i + 1 < args.size(); i += 2) r += " WHEN " + args[i]->str() + " THEN " + args[i + 1]->str();
      if (args.size() % 2) r += " ELSE " + args.back()->str();
      return r + " END";
  }
  return "?";
}

// ---------------------------------------------------------------- lexer

namespace {

struct Tok {
  enum K { ID, QID, NUM, STR, SYM, END } k;
  std::string v;
};

std::vector<Tok> lex(const std::string& s) {
  std::vector<Tok> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    char c = s[i];
    if (isspace((unsigned char)c)) { ++i; continue; }
    if (c == '-' && i + 1 < n && s[i + 1] == '-') { while (i < n && s[i] != '\n') ++i; continue; }
    if (c == '#') { while (i < n && s[i] != '\n') ++i; continue; }
    if (c == '/' && i + 1 < n && s[i + 1] == '*') {
      size_t e = s.find("*/", i + 2);
      i = e == std::string::npos ? n : e + 2;
      continue;
    }
    if (isalpha((unsigned char)c) || c == '_' || c == '@') {
      size_t j = i;
      while (j < n && (isalnum((unsigned char)s[j]) || s[j] == '_' || s[j] == '@' || s[j] == '$')) ++j;
      out.push_back({Tok::ID, s.substr(i, j - i)});
      i = j;
      continue;
    }
    if (isdigit((unsigned char)c) || (c == '.' && i + 1 < n && isdigit((unsigned char)s[i + 1]))) {
      size_t j = i;
      while (j < n && (isdigit((unsigned char)s[j]) || s[j] == '.')) ++j;
      if (j < n && (s[j] == 'e' || s[j] == 'E')) {
        ++j;
        if (j < n && (s[j] == '+' || s[j] == '-')) ++j;
        while (j < n && isdigit((unsigned char)s[j])) ++j;
      }
      out.push_back({Tok::NUM, s.substr(i, j - i)});
      i = j;
      continue;
    }
    if (c == '\'' || c == '"' || c == '`') {
      char q = c;
      std::string v;
      ++i;
      while (i < n) {
        if (s[i] == '\\' && q != '`' && i + 1 < n) {
          char e = s[i + 1];
          v += e == 'n' ? '\n' : e == 't' ? '\t' : e == '0' ? '\0' : e;
          i += 2;
        } else if (s[i] == q) {
          if (i + 1 < n && s[i + 1] == q) { v += q; i += 2; }
          else { ++i; break; }
        } else {
          v += s[i++];
        }
      }
      out.push_back({q == '`' ? Tok::QID : Tok::STR, v});
      continue;
    }
    static const char* three[] = {"<*>", "<->", "->>"};
    bool matched = false;
    for (auto t : three)
      if (s.compare(i, 3, t) == 0) { out.push_back({Tok::SYM, t}); i += 3; matched = true; break; }
    if (matched) continue;
    static const char* two[] = {"<=", ">=", "<>", "!=", "||", "&&", ":=", "->", "::"};
    for (auto t : two)
      if (s.compare(i, 2, t) == 0) { out.push_back({Tok::SYM, t}); i += 2; matched = true; break; }
    if (matched) continue;
    out.push_back({Tok::SYM, std::string(1, c)});
    ++i;
  }
  out.push_back({Tok::END, ""});
  return out;
}

const char* kReserved[] = {"FROM", "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "JOIN", "INNER", "ON", "AS",
                           "AND", "OR", "NOT", "LEFT", "RIGHT", "CROSS", "UNION", "BY", "ASC", "DESC", "SET",
                           "VALUES", "INTO", "OFFSET", "WHEN", "THEN", "ELSE", "END", "BETWEEN", "IN", "LIKE",
                           "IS", "SELECT", "STRAIGHT_JOIN", "USING", "FULL", "OUTER", "NATURAL", "WITH", "EXCEPT",
                           "INTERSECT", "WINDOW", "OVER", "PARTITION", nullptr};

struct Parser {
  std::vector<Tok> t;
  size_t p = 0;

  const Tok& cur() const { return t[p]; }
  bool isKw(const char* kw, size_t off = 0) const {
    const Tok& x = t[std::min(p + off, t.size() - 1)];
    return x.k == Tok::ID && strcasecmp(x.v.c_str(), kw) == 0;
  }
  bool isSym(const char* s) const { return cur().k == Tok::SYM && cur().v == s; }
  bool acceptKw(const char* kw) { if (isKw(kw)) { ++p; return true; } return false; }
  bool acceptSym(const char* s) { if (isSym(s)) { ++p; return true; } return false; }
  [[noreturn]] void fail(const std::string& what) {
    throw SqlError("You have an error in your SQL syntax: expected " + what + " near '" + cur().v + "'");
  }
  void expectKw(const char* kw) { if (!acceptKw(kw)) fail(kw); }
  void expectSym(const char* s) { if (!acceptSym(s)) fail(std::string("'") + s + "'"); }
  bool reserved(const Tok& x) const {
    if (x.k != Tok::ID) return false;
    for (int i = 0; kReserved[i]; ++i)
      if (strcasecmp(x.v.c_str(), kReserved[i]) == 0) return true;
    return false;
  }
  std::string ident() {
    if (cur().k == Tok::QID || (cur().k == Tok::ID && !reserved(cur()))) return t[p++].v;
    fail("identifier");
  }
  int64_t intLit() {
    if (cur().k != Tok::NUM) fail("number");
    return std::stoll(t[p++].v);
  }

  // ---- expressions
  ExprP expr() { return orExpr(); }
  ExprP orExpr() {
    ExprP l = andExpr();
    while (acceptKw("OR") || acceptSym("||")) l = mkBin("OR", l, andExpr());
    return l;
  }
  ExprP andExpr() {
    ExprP l = notExpr();
    while (acceptKw("AND") || acceptSym("&&")) l = mkBin("AND", l, notExpr());
    return l;
  }
  ExprP notExpr() {
    if (acceptKw("NOT") || acceptSym("!")) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::UNARY;
      e->op = "NOT";
      e->args = {notExpr()};
      return e;
    }
    return cmpExpr();
  }
  ExprP cmpExpr() {
    ExprP l = addExpr();
    for (;;) {
      static const char* ops[] = {"=", "<>", "!=", "<=", ">=", "<", ">"};
      bool done = true;
      for (auto o : ops)
        if (acceptSym(o)) {
          l = mkBin(std::string(o) == "!=" ? "<>" : o, l, addExpr());
          done = false;
          break;
        }
      if (!done) continue;
      if (acceptKw("IS")) {  // IS [NOT] NULL / TRUE / FALSE
        bool n = acceptKw("NOT");
        auto e = std::make_shared<Expr>();
        e->kind = Expr::FUNC;
        if (acceptKw("NULL")) {
          e->op = n ? "ISNOTNULL" : "ISNULL";
          e->args = {l};
        } else {
          bool tv = acceptKw("TRUE");
          if (!tv) expectKw("FALSE");
          e->op = (tv != n) ? "ISTRUE" : "ISFALSE";
          e->args = {l};
        }
        l = e;
        continue;
      }
      bool neg = false;
      size_t save = p;
      if (acceptKw("NOT")) neg = true;
      if (acceptKw("BETWEEN")) {
        auto e = std::make_shared<Expr>();
        e->kind = Expr::BETWEEN;
        e->neg = neg;
        ExprP lo = addExpr();
        expectKw("AND");
        e->args = {l, lo, addExpr()};
        l = e;
        continue;
      }
      if (acceptKw("IN")) {
        auto e = std::make_shared<Expr>();
        e->kind = Expr::INLIST;
        e->neg = neg;
        e->args = {l};
        expectSym("(");
        if (isKw("SELECT") || isKw("WITH")) {
          e->sub = std::make_shared<Stmt>(query());
        } else {
          do e->args.push_back(expr()); while (acceptSym(","));
        }
        expectSym(")");
        l = e;
        continue;
      }
      if (acceptKw("LIKE")) {
        auto e = std::make_shared<Expr>();
        e->kind = Expr::LIKE;
        e->neg = neg;
        e->args = {l, addExpr()};
        l = e;
        continue;
      }
      p = save;
      return l;
    }
  }
  ExprP addExpr() {
    ExprP l = mulExpr();
    for (;;) {
      if (acceptSym("+")) l = mkBin("+", l, mulExpr());
      else if (acceptSym("-")) l = mkBin("-", l, mulExpr());
      else return l;
    }
  }
  ExprP mulExpr() {
    ExprP l = unary();
    for (;;) {
      if (acceptSym("*")) l = mkBin("*", l, unary());
      else if (acceptSym("/")) l = mkBin("/", l, unary());
      else if (acceptSym("%") || acceptKw("MOD")) l = mkBin("%", l, unary());
      else if (acceptKw("DIV")) {  // integer division
        auto f = std::make_shared<Expr>();
        f->kind = Expr::FUNC;
        f->op = "DIV";
        f->args = {l, unary()};
        l = f;
      }
      else if (acceptSym("<*>") || acceptSym("<->")) {  // SingleStore vector operators
        auto e = std::make_shared<Expr>();
        e->kind = Expr::FUNC;
        e->op = t[p - 1].v == "<*>" ? "DOT_PRODUCT" : "EUCLIDEAN_DISTANCE";
        e->args = {l, unary()};
        l = e;
      }
      else return l;
    }
  }
  ExprP unary() {
    if (acceptSym("-")) {
      ExprP a = unary();
      if (a->kind == Expr::LIT && a->lit.kind == Value::INT) return mkLit(Value::I(-a->lit.i));
      if (a->kind == Expr::LIT && a->lit.kind == Value::DBL) return mkLit(Value::D(-a->lit.d));
      auto e = std::make_shared<Expr>();
      e->kind = Expr::UNARY;
      e->op = "-";
      e->args = {a};
      return e;
    }
    if (acceptSym("+")) return unary();
    if (isKw("BINARY") && t[p + 1].k != Tok::END && !(t[p + 1].k == Tok::SYM && (t[p + 1].v == "," || t[p + 1].v == ")"))) {
      ++p;  // BINARY expr: a binary-collation cast; comparisons here are already exact unless ci collation is set
      return unary();
    }
    ExprP e = primary();
    while (isSym("::")) {  // SingleStore JSON access: j::key (JSON), j::$key (string), j::%key (double)
      ++p;
      std::string fn = "JSON_EXTRACT_JSON";
      if (acceptSym("$")) fn = "JSON_EXTRACT_STRING";
      else if (acceptSym("%")) fn = "JSON_EXTRACT_DOUBLE";
      std::string key = cur().k == Tok::NUM ? t[p++].v : ident();
      auto f = std::make_shared<Expr>();
      f->kind = Expr::FUNC;
      f->op = fn;
      f->args = {e, mkLit(Value::S(key))};
      e = f;
    }
    while (isSym("->") || isSym("->>")) {  // MySQL JSON shorthands: col->'$.a', col->>'$.a'
      bool unq = t[p].v == "->>";
      ++p;
      auto f = std::make_shared<Expr>();
      f->kind = Expr::FUNC;
      f->op = "JSON_EXTRACT";
      f->args = {e, primary()};
      if (unq) {
        auto u = std::make_shared<Expr>();
        u->kind = Expr::FUNC;
        u->op = "JSON_UNQUOTE";
        u->args = {f};
        f = u;
      }
      e = f;
    }
    return e;
  }
  ExprP primary() {
    const Tok& x = cur();
    if (x.k == Tok::NUM) {
      ++p;
      if (x.v.find_first_of("eE") != std::string::npos) return mkLit(Value::D(std::stod(x.v)));
      if (auto dot = x.v.find('.'); dot != std::string::npos)  // 1.25 is an exact DECIMAL(.., 2) literal (MySQL)
        return mkLit(Value::Dec(std::stod(x.v), (int)std::min<size_t>(30, x.v.size() - dot - 1)));
      return mkLit(Value::I(std::stoll(x.v)));
    }
    if (x.k == Tok::STR) { ++p; return mkLit(Value::S(x.v)); }
    if (acceptSym("(")) {
      if (isKw("SELECT") || isKw("WITH")) {
        auto e = std::make_shared<Expr>();
        e->kind = Expr::SUBQUERY;
        e->sub = std::make_shared<Stmt>(query());
        expectSym(")");
        return e;
      }
      ExprP e = expr();
      expectSym(")");
      return e;
    }
    if (isKw("MATCH") && t[p + 1].k == Tok::SYM && t[p + 1].v == "(") {  // MATCH (cols) AGAINST ('text' [IN BOOLEAN MODE])
      p += 2;
      auto e = std::make_shared<Expr>();
      e->kind = Expr::FUNC;
      e->op = "MATCH_AGAINST";
      do e->args.push_back(expr()); while (acceptSym(","));
      expectSym(")");
      expectKw("AGAINST");
      expectSym("(");
      e->args.push_back(primary());  // not expr(): "'text' IN BOOLEAN MODE" is not an IN list
      if (acceptKw("IN")) {
        if (acceptKw("BOOLEAN")) e->castType = "BOOLEAN";
        else { expectKw("NATURAL"); expectKw("LANGUAGE"); }
        expectKw("MODE");
      }
      if (acceptKw("WITH")) { expectKw("QUERY"); expectKw("EXPANSION"); }
      expectSym(")");
      return e;
    }
    if (isKw("EXISTS") && t[p + 1].k == Tok::SYM && t[p + 1].v == "(") {
      p += 2;
      auto e = std::make_shared<Expr>();
      e->kind = Expr::FUNC;
      e->op = "EXISTS";
      e->sub = std::make_shared<Stmt>(query());
      expectSym(")");
      return e;
    }
    if (acceptSym("*")) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::STAR;
      return e;
    }
    if (x.k == Tok::ID && x.v.rfind("@@", 0) == 0) {
      std::string v = upper(x.v);
      ++p;
      if (acceptSym(".")) v = "@@" + upper(ident());
      std::string name = v.substr(2);  // @@[session. | global. | local.]name
      for (const char* pre : {"SESSION.", "GLOBAL.", "LOCAL."})
        if (name.rfind(pre, 0) == 0) name = name.substr(strlen(pre));
      auto e = mkLit(sysVarValue(name));
      e->name = "@@" + x.v.substr(2);  // the column name clients see
      return e;
    }
    if (isKw("DATE") && t[p + 1].k == Tok::STR) {
      p++;
      int32_t d;
      if (!parseDate(t[p].v, d)) fail("valid date");
      ++p;
      return mkLit(Value::Dt(d));
    }
    if (acceptKw("INTERVAL")) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::INTERVAL;
      if (cur().k == Tok::STR) e->lit = Value::I(std::stoll(t[p++].v));
      else e->lit = Value::I(intLit());
      std::string u = upper(ident());
      if (u.size() > 1 && u.back() == 'S') u.pop_back();
      static const char* units[] = {"MICROSECOND", "SECOND", "MINUTE", "HOUR", "DAY", "WEEK", "MONTH", "QUARTER", "YEAR"};
      bool ok = false;
      for (auto x : units) ok |= u == x;
      if (!ok) fail("interval unit");
      e->op = u;
      return e;
    }
    if (acceptKw("CASE")) {
      auto e = std::make_shared<Expr>();
      e->kind = Expr::CASE;
      ExprP subject;
      if (!isKw("WHEN")) subject = expr();
      while (acceptKw("WHEN")) {
        ExprP c = expr();
        if (subject) c = mkBin("=", subject, c);
        expectKw("THEN");
        e->args.push_back(c);
        e->args.push_back(expr());
      }
      if (acceptKw("ELSE")) e->args.push_back(expr());
      expectKw("END");
      return e;
    }
    if (acceptKw("NULL")) return mkLit(Value::Null());
    if (isKw("TRUE")) { ++p; return mkLit(Value::I(1)); }
    if (isKw("FALSE")) { ++p; return mkLit(Value::I(0)); }
    if (x.k == Tok::ID || x.k == Tok::QID) {
      std::string name = t[p++].v;
      if (x.k == Tok::ID && isSym("(")) {
        ++p;
        auto e = std::make_shared<Expr>();
        e->kind = Expr::FUNC;
        e->op = upper(name);
        if (e->op == "EXTRACT") {  // EXTRACT(unit FROM x) -> unit(x)
          std::string unit = upper(ident());
          expectKw("FROM");
          e->op = unit == "DAY" ? "DAYOFMONTH" : unit;
          e->args.push_back(expr());
          expectSym(")");
          return e;
        }
        if (e->op == "CAST" || e->op == "CONVERT") {  // CAST(x AS type), CONVERT(x, type)
          e->args.push_back(expr());
          if (e->op == "CAST") expectKw("AS");
          else expectSym(",");
          e->castType = upper(ident());
          if (acceptSym("(")) { while (!acceptSym(")")) ++p; }
          if (e->castType == "UNSIGNED" || e->castType == "SIGNED") acceptKw("INTEGER") || acceptKw("INT");
          e->op = "CAST";
          expectSym(")");
          return e;
        }
        if ((e->op == "SUBSTRING" || e->op == "SUBSTR") && !isSym(")")) {  // SUBSTRING(x FROM a [FOR b])
          e->args.push_back(expr());
          if (acceptKw("FROM")) {
            e->args.push_back(expr());
            if (acceptKw("FOR")) e->args.push_back(expr());
            expectSym(")");
            return e;
          }
          while (acceptSym(",")) e->args.push_back(expr());
          expectSym(")");
          return e;
        }
        if (e->op == "TIMESTAMPDIFF" || e->op == "TIMESTAMPADD") {  // first argument is a unit keyword
          std::string unit = upper(ident());
          if (unit.size() > 1 && unit.back() == 'S') unit.pop_back();
          e->castType = unit;
          expectSym(",");
          do e->args.push_back(expr()); while (acceptSym(","));
          expectSym(")");
          return e;
        }
        if (acceptKw("DISTINCT")) e->distinct = true;
        if (e->op == "APPROX_COUNT_DISTINCT") e->op = "COUNT", e->distinct = true;  // exact here
        if (!isSym(")")) do e->args.push_back(expr()); while (acceptSym(","));
        if (acceptKw("ORDER")) {  // GROUP_CONCAT(x ORDER BY y [DESC] ..)
          expectKw("BY");
          do {
            ExprP o = expr();
            bool desc = acceptKw("DESC");
            if (!desc) acceptKw("ASC");
            e->winOrder.push_back({o, desc});
          } while (acceptSym(","));
        }
        if (acceptKw("SEPARATOR")) {
          if (cur().k != Tok::STR) fail("separator string");
          e->name = t[p++].v;
          e->castType = "SEP";
        }
        expectSym(")");
        if (isKw("WITHIN") && upper(t[p + 1].v) == "GROUP") {  // PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY x)
          p += 2;
          expectSym("(");
          expectKw("ORDER");
          expectKw("BY");
          ExprP x = expr();
          if (acceptKw("DESC")) {
            auto one = mkLit(Value::D(1.0));
            if (!e->args.empty()) e->args[0] = mkBin("-", one, e->args[0]);
          } else {
            acceptKw("ASC");
          }
          expectSym(")");
          e->args.insert(e->args.begin(), x);  // -> PERCENTILE_CONT(x, p)
        }
        if (acceptKw("OVER")) {
          e->window = true;
          expectSym("(");
          if (acceptKw("PARTITION")) {
            expectKw("BY");
            do e->partBy.push_back(expr()); while (acceptSym(","));
          }
          if (acceptKw("ORDER")) {
            expectKw("BY");
            do {
              ExprP o = expr();
              bool desc = acceptKw("DESC");
              if (!desc) acceptKw("ASC");
              e->winOrder.push_back({o, desc});
            } while (acceptSym(","));
          }
          if (isKw("ROWS") || isKw("RANGE")) {
            bool rows = acceptKw("ROWS");
            if (!rows) expectKw("RANGE");
            auto bound = [&](bool lo) -> int64_t {
              if (acceptKw("UNBOUNDED")) {
                if (acceptKw("PRECEDING")) return INT64_MIN;
                expectKw("FOLLOWING");
                return INT64_MAX;
              }
              if (acceptKw("CURRENT")) { expectKw("ROW"); return 0; }
              int64_t n = intLit();
              if (acceptKw("PRECEDING")) return -n;
              expectKw("FOLLOWING");
              (void)lo;
              return n;
            };
            if (acceptKw("BETWEEN")) {
              e->frameLo = bound(true);
              expectKw("AND");
              e->frameHi = bound(false);
            } else {
              e->frameLo = bound(true);
              e->frameHi = 0;
            }
            e->rowsFrame = rows;
          }
          expectSym(")");
        }
        if (e->op == "DATABASE" || e->op == "SCHEMA") return mkLit(currentDb().empty() ? Value::Null() : Value::S(currentDb()));
        if (e->op == "VERSION") return mkLit(Value::S("8.0.32-gpudb"));
        if (e->op == "USER" || e->op == "SESSION_USER" || e->op == "SYSTEM_USER") return mkLit(Value::S(sessionUser() + "@localhost"));
        if (e->op == "CURRENT_USER") return mkLit(Value::S(sessionUser()));
        if (e->op == "CONNECTION_ID") return mkLit(Value::I(tSession ? tSession->id : 1));
        if (udfHook && !e->window)
          if (ExprP u = udfHook(e)) return u;
        return e;
      }
      if (acceptSym(".")) {
        if (acceptSym("*")) {
          auto e = std::make_shared<Expr>();
          e->kind = Expr::STAR;
          e->table = name;
          return e;
        }
        std::string second = ident();
        if (acceptSym(".")) {  // db.table.column (or db.table.*): the table qualifies it
          if (acceptSym("*")) {
            auto e = std::make_shared<Expr>();
            e->kind = Expr::STAR;
            e->table = second;
            return e;
          }
          return mkCol(second, ident());
        }
        return mkCol(name, second);
      }
      if (name.size() > 1 && name[0] == '@' && name[1] != '@') return mkLit(userVar(name));  // @user variable
      return mkCol("", name);
    }
    fail("expression");
  }

  // ---- statements
  void tableRefs(Stmt& s) {
    auto one = [&](int join) {
      TableRef r;
      r.join = join;
      r.lateral = acceptKw("LATERAL");
      if (acceptSym("(")) {  // derived table
        r.sub = std::make_shared<Stmt>(query());
        expectSym(")");
        r.name = "";
      } else {
        r.name = tableName();
        if (acceptSym("(")) {  // table-valued function: FROM fn(args)
          r.tvf = true;
          if (!acceptSym(")")) {
            do r.tvfArgs.push_back(expr()); while (acceptSym(","));
            expectSym(")");
          }
        }
      }
      if (acceptKw("AS")) r.alias = ident();
      else if ((cur().k == Tok::ID && !reserved(cur())) || cur().k == Tok::QID) r.alias = ident();
      if (r.sub && !r.alias.empty() && isSym("(")) {  // derived table column names: AS t (a, b, ..)
        auto names = parenIdents();
        for (size_t i = 0; i < names.size() && i < r.sub->items.size(); ++i) r.sub->items[i].alias = names[i];
      }
      if (r.alias.empty()) {
        if (r.sub) throw SqlError("Every derived table must have its own alias", 1248);
        r.alias = r.name.substr(r.name.find('.') == std::string::npos ? 0 : r.name.find('.') + 1);
      }
      s.from.push_back(r);
    };
    one(0);
    for (;;) {
      if (acceptSym(",")) { one(0); continue; }
      int join = -1;
      const bool natural = acceptKw("NATURAL");
      if (acceptKw("INNER") || acceptKw("CROSS")) { expectKw("JOIN"); join = 0; }
      else if (acceptKw("JOIN") || acceptKw("STRAIGHT_JOIN")) join = 0;
      else if (acceptKw("LEFT")) { acceptKw("OUTER"); expectKw("JOIN"); join = 1; }
      else if (acceptKw("RIGHT")) { acceptKw("OUTER"); expectKw("JOIN"); join = 2; }
      else if (acceptKw("FULL")) { acceptKw("OUTER"); expectKw("JOIN"); join = 3; }
      if (join < 0) {
        if (natural) throw SqlError("You have an error in your SQL syntax: expected JOIN after NATURAL", 1064);
        break;
      }
      one(join);
      s.from.back().natural = natural;
      ExprP cond;
      if (acceptKw("ON")) {
        cond = expr();
      } else if (acceptKw("USING")) {
        const std::string left = s.from[s.from.size() - 2].alias, right = s.from.back().alias;
        for (auto& c : parenIdents()) {
          s.from.back().usingCols.push_back(c);
          ExprP eq = mkBin("=", mkCol(left, c), mkCol(right, c));
          cond = cond ? mkBin("AND", cond, eq) : eq;
        }
      }
      if (cond) {
        if (join == 0) s.joinConds.push_back(cond);
        else s.from.back().on = cond;
      }
    }
  }

  // A full query: [WITH ..] SELECT .. [UNION [ALL] SELECT ..]* [ORDER BY .. LIMIT ..]
  Stmt query() {
    std::vector<std::pair<std::string, std::shared_ptr<Stmt>>> ctes;
    if (acceptKw("WITH")) {
      acceptKw("RECURSIVE");
      do {
        std::string name = ident();
        std::vector<std::string> colNames;
        if (isSym("(")) colNames = parenIdents();
        expectKw("AS");
        expectSym("(");
        auto q = std::make_shared<Stmt>(query());
        expectSym(")");
        if (!colNames.empty())  // WITH t(a, b) AS (..): rename the output columns
          for (size_t i = 0; i < colNames.size() && i < q->items.size(); ++i) q->items[i].alias = colNames[i];
        ctes.push_back({name, q});
      } while (acceptSym(","));
    }
    Stmt s = selectOrParen();
    Stmt* last = &s;
    while (isKw("UNION") || isKw("INTERSECT") || isKw("EXCEPT") || isKw("MINUS")) {
      const std::string op = upper(t[p++].v);
      bool all = acceptKw("ALL");
      if (!all) acceptKw("DISTINCT");
      last->unionAll = all;
      last->setOp = op == "UNION" ? 0 : op == "INTERSECT" ? 1 : 2;
      last->unionNext = std::make_shared<Stmt>(selectOrParen());
      last = last->unionNext.get();
    }
    if (last != &s) {  // trailing ORDER BY / LIMIT belong to the whole UNION
      if (!last->orderBy.empty() || last->limit >= 0) {
        s.orderBy = last->orderBy;
        s.limit = last->limit;
        s.offset = last->offset;
        last->orderBy.clear();
        last->limit = -1;
        last->offset = 0;
        s.unionOrder = true;
      }
      orderLimit(s);
    }
    s.ctes = std::move(ctes);
    return s;
  }
  Stmt selectOrParen() {
    if (acceptSym("(")) {
      Stmt q = query();
      expectSym(")");
      return q;
    }
    return select();
  }
  void orderLimit(Stmt& s) {
    if (acceptKw("ORDER")) {
      expectKw("BY");
      s.orderBy.clear();
      do {
        OrderItem o;
        o.e = expr();
        if (acceptKw("DESC")) o.desc = true;
        else acceptKw("ASC");
        s.orderBy.push_back(o);
      } while (acceptSym(","));
      s.unionOrder = s.unionNext != nullptr;
    }
    if (acceptKw("LIMIT")) {
      int64_t a = intLit();
      if (acceptSym(",")) { s.offset = a; s.limit = intLit(); }
      else {
        s.limit = a;
        if (acceptKw("OFFSET")) s.offset = intLit();
      }
    }
  }

  Stmt select() {
    Stmt s;
    s.type = Stmt::SELECT;
    expectKw("SELECT");
    if (acceptKw("DISTINCT")) s.distinct = true;
    else acceptKw("ALL");
    do {
      SelectItem it;
      it.e = expr();
      if (acceptKw("AS")) it.alias = cur().k == Tok::STR ? t[p++].v : ident();
      else if ((cur().k == Tok::ID && !reserved(cur())) || cur().k == Tok::QID) it.alias = ident();
      else if (cur().k == Tok::STR) it.alias = t[p++].v;
      s.items.push_back(it);
    } while (acceptSym(","));
    intoOutfile(s);  // SELECT .. INTO OUTFILE 'f' FROM ..
    if (acceptKw("FROM")) tableRefs(s);
    if (acceptKw("WHERE")) s.where = expr();
    if (acceptKw("GROUP")) {
      expectKw("BY");
      // GROUP BY a, b [WITH ROLLUP] | ROLLUP(a, b) | CUBE(a, b) | GROUPING SETS ((a, b), (a), ())
      auto indexOf = [&](const ExprP& e) {
        for (size_t i = 0; i < s.groupBy.size(); ++i)
          if (s.groupBy[i]->str() == e->str()) return (int)i;
        s.groupBy.push_back(e);
        return (int)s.groupBy.size() - 1;
      };
      auto rollup = [&](const std::vector<int>& ks) {
        for (size_t n = ks.size() + 1; n-- > 0;) s.groupingSets.push_back(std::vector<int>(ks.begin(), ks.begin() + n));
      };
      if (isKw("ROLLUP") && t[p + 1].v == "(") {
        ++p;
        expectSym("(");
        std::vector<int> ks;
        do ks.push_back(indexOf(expr())); while (acceptSym(","));
        expectSym(")");
        rollup(ks);
      } else if (isKw("CUBE") && t[p + 1].v == "(") {
        ++p;
        expectSym("(");
        std::vector<int> ks;
        do ks.push_back(indexOf(expr())); while (acceptSym(","));
        expectSym(")");
        for (int mask = (1 << ks.size()) - 1; mask >= 0; --mask) {
          std::vector<int> set;
          for (size_t i = 0; i < ks.size(); ++i)
            if (mask & (1 << i)) set.push_back(ks[i]);
          s.groupingSets.push_back(set);
        }
      } else if (isKw("GROUPING") && upper(t[p + 1].v) == "SETS") {
        p += 2;
        expectSym("(");
        do {
          std::vector<int> set;
          if (acceptSym("(")) {
            if (!acceptSym(")")) {
              do set.push_back(indexOf(expr())); while (acceptSym(","));
              expectSym(")");
            }
          } else {
            set.push_back(indexOf(expr()));
          }
          s.groupingSets.push_back(set);
        } while (acceptSym(","));
        expectSym(")");
      } else {
        do s.groupBy.push_back(expr()); while (acceptSym(","));
        if (isKw("WITH") && upper(t[p + 1].v) == "ROLLUP") {
          p += 2;
          std::vector<int> ks;
          for (size_t i = 0; i < s.groupBy.size(); ++i) ks.push_back((int)i);
          rollup(ks);
        }
      }
    }
    if (acceptKw("HAVING")) s.having = expr();
    orderLimit(s);
    intoOutfile(s);
    if (acceptKw("FOR")) {  // FOR UPDATE / FOR SHARE: locking reads (table locks already serialize)
      if (!acceptKw("UPDATE")) expectKw("SHARE");
      while (acceptKw("NOWAIT") || acceptKw("SKIP") || acceptKw("LOCKED")) {}
    }
    if (acceptKw("LOCK")) { expectKw("IN"); expectKw("SHARE"); expectKw("MODE"); }
    return s;
  }
  void intoOutfile(Stmt& s) {
    if (!isKw("INTO")) return;
    ++p;
    expectKw("OUTFILE");
    if (cur().k != Tok::STR) fail("file path");
    s.outfile = t[p++].v;
    for (;;) {
      if (acceptKw("FIELDS") || acceptKw("COLUMNS")) {
        while (acceptKw("TERMINATED") || acceptKw("OPTIONALLY") || acceptKw("ENCLOSED") || acceptKw("ESCAPED")) {
          if (t[p - 1].v == "OPTIONALLY" || strcasecmp(t[p - 1].v.c_str(), "OPTIONALLY") == 0) expectKw("ENCLOSED");
          bool term = strcasecmp(t[p - 1].v.c_str(), "TERMINATED") == 0;
          expectKw("BY");
          std::string d = t[p++].v;
          if (term) s.outDelim = d == "\\t" || d == "t" || d.empty() ? '\t' : d[0];
        }
      } else if (acceptKw("LINES")) {
        while (acceptKw("TERMINATED") || acceptKw("STARTING")) { expectKw("BY"); ++p; }
      } else {
        break;
      }
    }
  }

  LType typeName(std::string* text = nullptr, int* vecDim = nullptr, int* scale = nullptr) {
    int nums[2] = {-1, -1}, nn = 0;
    std::string tn = upper(ident());
    std::string txt = tn;
    if (tn == "DOUBLE") acceptKw("PRECISION");
    if (tn == "CHARACTER" && acceptKw("VARYING")) tn = "VARCHAR";
    if (acceptSym("(")) {  // VARCHAR(25), DECIMAL(15,2), ENUM('a','b'), VECTOR(4)
      txt += "(";
      bool first = true;
      while (!isSym(")") && cur().k != Tok::END) {
        if (!first && !isSym(",")) txt += "";
        const Tok& k = t[p++];
        txt += k.k == Tok::STR ? "'" + k.v + "'" : k.v;
        if (k.k == Tok::NUM && nn < 2) nums[nn++] = std::atoi(k.v.c_str());
        if (tn == "VECTOR" && first && vecDim) *vecDim = std::atoi(k.v.c_str());
        first = false;
      }
      expectSym(")");
      txt += ")";
    }
    while (acceptKw("UNSIGNED") || acceptKw("SIGNED") || acceptKw("ZEROFILL")) {}
    if (acceptKw("CHARACTER")) { expectKw("SET"); ident(); }
    if (acceptKw("CHARSET")) ident();
    if (acceptKw("COLLATE")) ident();
    if (text) *text = txt;
    if (tn == "INT" || tn == "INTEGER" || tn == "SMALLINT" || tn == "TINYINT" || tn == "MEDIUMINT" || tn == "BOOL" ||
        tn == "BOOLEAN" || tn == "BIT" || tn == "YEAR")
      return LType::INT;
    if (tn == "BIGINT" || tn == "SERIAL") return LType::BIGINT;
    if (tn == "DECIMAL" || tn == "NUMERIC" || tn == "DEC" || tn == "FIXED") {  // exact: scaled int64
      int sc = nums[1] >= 0 ? nums[1] : 0;
      if (nums[0] > 18 || sc > 9) return LType::DOUBLE;  // beyond int64 fixed point: approximate
      if (scale) *scale = sc;
      return LType::DECIMAL;
    }
    if (tn == "DOUBLE" || tn == "FLOAT" || tn == "REAL") return LType::DOUBLE;
    if (tn == "DATE") return LType::DATE;
    if (tn == "DATETIME" || tn == "TIMESTAMP") return LType::DATETIME;
    if (tn == "VARCHAR" || tn == "CHAR" || tn == "TEXT" || tn == "STRING" || tn == "LONGTEXT" || tn == "MEDIUMTEXT" ||
        tn == "TINYTEXT" || tn == "JSON" || tn == "ENUM" || tn == "SET" || tn == "TIME" || tn == "BLOB" ||
        tn == "LONGBLOB" || tn == "MEDIUMBLOB" || tn == "TINYBLOB" || tn == "BINARY" || tn == "VARBINARY" ||
        tn == "NVARCHAR" || tn == "NCHAR" || tn == "GEOGRAPHY" || tn == "GEOGRAPHYPOINT")
      return LType::VARCHAR;
    if (tn == "VECTOR") return LType::VECTOR;
    throw SqlError("unsupported column type " + tn);
  }

  // KEY / INDEX / UNIQUE [KEY|INDEX] [name] (cols) [USING ...]: secondary index definitions.
  // `KEY (c) USING CLUSTERED COLUMNSTORE` is SingleStore's older sort-key syntax.
  void indexDef(Stmt& s, bool unique) {
    IndexDef ix;
    ix.unique = unique;
    if (cur().k == Tok::ID || cur().k == Tok::QID) ix.name = ident();
    if (acceptKw("USING")) ident();
    ix.cols = parenIdents();
    bool columnstore = false;
    while (!isSym(",") && !isSym(")") && cur().k != Tok::END) {
      if (isKw("CLUSTERED") || isKw("COLUMNSTORE")) columnstore = true;
      if (acceptKw("INDEX_OPTIONS")) {
        if (cur().k == Tok::STR) ix.options = t[p++].v;
        continue;
      }
      ++p;
    }
    if (columnstore) s.sortKey = ix.cols.at(0);
    else s.indexes.push_back(ix);
  }

  std::string parenIdent() { return parenIdents().at(0); }
  // 'user'@'host' / user@host / user -> user name
  std::string userName() {
    std::string u = cur().k == Tok::STR || cur().k == Tok::QID || cur().k == Tok::ID ? t[p++].v : "";
    if (u.rfind("@", 0) != 0 && u.find('@') != std::string::npos) u = u.substr(0, u.find('@'));
    if (acceptSym("@")) {
      ++p;  // host
    } else if (cur().k == Tok::ID && cur().v.rfind("@", 0) == 0) {
      bool bare = cur().v.size() == 1;  // "@" then a quoted host
      ++p;
      if (bare && (cur().k == Tok::STR || cur().k == Tok::ID || cur().k == Tok::QID)) ++p;
    }
    return u;
  }
  std::string identifiedBy() {  // IDENTIFIED [WITH plugin] BY 'pw'
    if (!acceptKw("IDENTIFIED")) return "\x01";
    if (acceptKw("WITH")) ++p;
    expectKw("BY");
    if (cur().k != Tok::STR) fail("password string");
    return t[p++].v;
  }
  // [db.]table -> "db.table" or "table"
  std::string tableName() {
    std::string n = ident();
    if (acceptSym(".")) n += "." + ident();
    return n;
  }
  std::vector<std::string> parenIdents(bool allowEmpty = false) {
    expectSym("(");
    std::vector<std::string> v;
    if (allowEmpty && acceptSym(")")) return v;
    do v.push_back(ident()); while (acceptSym(","));
    expectSym(")");
    return v;
  }

  // LOAD DATA [LOCAL] {INFILE | FS | S3 | KAFKA} 'path' [REPLACE | IGNORE] INTO TABLE t
  //   [FIELDS TERMINATED BY 'x' [OPTIONALLY] ENCLOSED BY 'c' ESCAPED BY 'c'] [LINES TERMINATED BY ..]
  //   [IGNORE n LINES] [(col | @skip, ..)]
  Stmt loadData() {
    Stmt s;
    s.type = Stmt::LOAD;
    expectKw("LOAD");
    expectKw("DATA");
    if (acceptKw("LOCAL")) s.setName = "LOCAL";
    if (acceptKw("INFILE")) s.setValue = "FS";
    else if (acceptKw("FS")) s.setValue = "FS";
    else if (acceptKw("S3") || acceptKw("KAFKA") || acceptKw("GCS") || acceptKw("AZURE") || acceptKw("HDFS") || acceptKw("LINK") || acceptKw("MYSQL"))
      s.setValue = upper(t[p - 1].v);
    else fail("INFILE / FS");
    if (cur().k != Tok::STR) fail("file path");
    s.path = t[p++].v;
    for (;;) {  // CONFIG / CREDENTIALS / BATCH_INTERVAL / REPLACE / IGNORE / SKIP .. ERRORS
      if (acceptKw("CONFIG")) {
        if (cur().k == Tok::STR) s.config = t[p++].v;
      } else if (acceptKw("CREDENTIALS")) {
        if (cur().k == Tok::STR) s.credentials = t[p++].v;
      } else if (cur().k == Tok::ID && !isKw("INTO")) {
        ++p;
      } else if (cur().k == Tok::STR || cur().k == Tok::NUM) {
        ++p;
      } else {
        break;
      }
    }
    expectKw("INTO");
    if (s.setValue == "MYSQL" && (acceptKw("DATABASE") || acceptKw("SCHEMA"))) {  // LOAD DATA MYSQL 'url' INTO DATABASE db
      s.table = ident();
      s.format = "DATABASE";
      while (cur().k != Tok::END && !isSym(";")) ++p;
      return s;
    }
    expectKw("TABLE");
    s.table = tableName();
    for (;;) {
      if (acceptKw("FIELDS") || acceptKw("COLUMNS")) {
        for (;;) {
          if (acceptKw("TERMINATED")) {
            expectKw("BY");
            if (cur().k != Tok::STR) fail("delimiter");
            std::string d = t[p++].v;
            s.delim = d == "\\t" || d == "t" ? '\t' : d.empty() ? '\t' : d[0];
          } else if (acceptKw("OPTIONALLY") || isKw("ENCLOSED")) {
            expectKw("ENCLOSED");
            expectKw("BY");
            ++p;
          } else if (acceptKw("ESCAPED")) {
            expectKw("BY");
            if (cur().k == Tok::STR && !t[p].v.empty()) s.escape = t[p].v[0];
            ++p;
          } else {
            break;
          }
        }
      } else if (acceptKw("LINES")) {
        while (acceptKw("TERMINATED") || acceptKw("STARTING")) {
          expectKw("BY");
          ++p;
        }
      } else if (acceptKw("FORMAT")) {
        s.format = upper(ident());
      } else if (acceptKw("IGNORE")) {
        s.skipLines = intLit();
        if (!acceptKw("LINES")) expectKw("ROWS");
      } else if (isSym("(")) {
        expectSym("(");
        do {
          if (acceptSym("@")) s.insertCols.push_back("@" + ident());
          else if (cur().k == Tok::ID && cur().v.rfind("@", 0) == 0) s.insertCols.push_back(t[p++].v);
          else s.insertCols.push_back(ident());
          std::string path = s.insertCols.back();
          if (acceptSym("<-") || (isSym("<") && t[p + 1].v == "-" && (p += 2))) {  // col <- key::sub
            path.clear();
            for (;;) {  // (JSON keys: any word, keywords too, or a quoted name)
              if (cur().k == Tok::ID || cur().k == Tok::QID || cur().k == Tok::STR) path += t[p++].v;
              else fail("JSON key");
              if (acceptSym("::")) { path += "::"; continue; }
              if (isSym(":") && t[p + 1].v == ":") { p += 2; path += "::"; continue; }
              break;
            }
          }
          s.jsonPaths.push_back(path);
        } while (acceptSym(","));
        expectSym(")");
      } else {
        break;
      }
    }
    while (cur().k != Tok::END && !isSym(";")) ++p;  // SET .., FORMAT .., etc.
    return s;
  }

  // KEY [index_name] (cols) REFERENCES t (cols) [ON DELETE a] [ON UPDATE a], after FOREIGN.
  ForeignKeyDef fkDef(const std::string& name) {
    ForeignKeyDef f;
    expectKw("KEY");
    f.name = name;
    if (cur().k == Tok::ID || cur().k == Tok::QID) {
      std::string ixName = ident();
      if (f.name.empty()) f.name = ixName;
    }
    f.cols = parenIdents();
    expectKw("REFERENCES");
    fkTail(f);
    return f;
  }
  void fkTail(ForeignKeyDef& f) {  // t (cols) [MATCH ..] [ON DELETE a] [ON UPDATE a]
    f.refTable = tableName();
    if (isSym("(")) f.refCols = parenIdents();
    if (acceptKw("MATCH")) ++p;
    while (acceptKw("ON")) {
      const bool del = acceptKw("DELETE");
      if (!del) expectKw("UPDATE");
      std::string a;
      if (acceptKw("CASCADE")) a = "CASCADE";
      else if (acceptKw("RESTRICT")) a = "RESTRICT";
      else if (acceptKw("SET")) a = acceptKw("NULL") ? "SET NULL" : (expectKw("DEFAULT"), "SET DEFAULT");
      else { expectKw("NO"); expectKw("ACTION"); a = "NO ACTION"; }
      (del ? f.onDelete : f.onUpdate) = a;
    }
  }

  // [=] AUTO | PIN | NEVER (VRAM placement hint)
  int vramHint() {
    acceptSym("=");
    if (acceptKw("PIN") || acceptKw("PINNED")) return 1;
    if (acceptKw("NEVER") || acceptKw("HOST")) return 2;
    if (acceptKw("AUTO") || acceptKw("DEFAULT")) return 0;
    throw SqlError("VRAM must be AUTO, PIN or NEVER", 1064);
  }

  ColumnDef columnDef(bool* isPk = nullptr) {
    ColumnDef c;
    c.name = ident();
    c.type = typeName(&c.typeText, &c.vecDim, &c.scale);
    while (!isSym(",") && !isSym(")") && cur().k != Tok::END && !isKw("FIRST") && !isKw("AFTER")) {
      if (acceptKw("PRIMARY")) { expectKw("KEY"); if (isPk) *isPk = true; c.notNull = true; }
      else if (acceptKw("NOT")) { expectKw("NULL"); c.notNull = true; }
      else if (acceptKw("NULL")) {}
      else if (acceptKw("AUTO_INCREMENT") || acceptKw("AUTOINCREMENT")) c.autoInc = true;
      else if (acceptKw("UNIQUE")) { acceptKw("KEY"); c.unique = true; }
      else if (acceptKw("DEFAULT")) {
        if (isKw("CURRENT_TIMESTAMP") || isKw("NOW") || isKw("LOCALTIMESTAMP")) {
          ++p;
          if (acceptSym("(")) { if (cur().k == Tok::NUM) ++p; expectSym(")"); }
          c.def = mkLit(Value::S("CURRENT_TIMESTAMP"));
        } else if (isSym("(")) {  // DEFAULT (expr): now() & co. are evaluated per row
          ++p;
          ExprP e = expr();
          expectSym(")");
          static const std::set<std::string> nowFns = {"NOW", "CURRENT_TIMESTAMP", "LOCALTIMESTAMP", "SYSDATE", "UTC_TIMESTAMP"};
          if (e->kind == Expr::FUNC && nowFns.count(upper(e->op)) && e->args.size() <= 1) {
            c.def = mkLit(Value::S("CURRENT_TIMESTAMP"));
            std::string fn = e->op;
            for (auto& ch : fn) ch = (char)tolower((unsigned char)ch);
            c.defText = fn + "()";
          } else {
            c.def = e;
          }
        } else {
          c.def = unary();
        }
      }
      else if (acceptKw("COMMENT") || acceptKw("COLLATE")) ++p;
      else if (acceptKw("VRAM")) c.vram = vramHint();
      else if (acceptKw("REFERENCES")) { c.ref = std::make_shared<ForeignKeyDef>(); fkTail(*c.ref); }
      else if (acceptKw("CHECK")) { expectSym("("); int depth = 1; while (depth && cur().k != Tok::END) { if (isSym("(")) ++depth; if (isSym(")")) --depth; ++p; } }
      else if (acceptKw("ON")) { expectKw("UPDATE"); ++p; if (acceptSym("(")) expectSym(")"); }
      else ++p;  // SERIES TIMESTAMP, SPARSE, ...
    }
    return c;
  }
  // Source text of tokens [a, b) (view definitions).
  std::string joinTokens(size_t a, size_t b) {
    std::string r;
    for (size_t i = a; i < b && i < t.size(); ++i) {
      if (t[i].k == Tok::END) break;
      if (!r.empty()) r += ' ';
      if (t[i].k == Tok::STR) {
        r += '\'';
        for (char ch : t[i].v) r += ch == '\'' ? std::string("''") : std::string(1, ch);
        r += '\'';
      } else if (t[i].k == Tok::QID) {
        r += "`" + t[i].v + "`";
      } else {
        r += t[i].v;
      }
    }
    return r;
  }

  Stmt create() {
    Stmt s;
    s.type = Stmt::CREATE;
    s.temporary = acceptKw("TEMPORARY");
    expectKw("TABLE");
    if (acceptKw("IF")) { expectKw("NOT"); expectKw("EXISTS"); s.ifNotExists = true; }
    s.table = tableName();
    if (acceptKw("LIKE")) {  // CREATE TABLE t LIKE other
      s.path = tableName();
      return s;
    }
    if (acceptKw("AS") || isKw("SELECT") || isKw("WITH")) {  // CREATE TABLE t AS SELECT ..
      s.select = std::make_shared<Stmt>(query());
      return s;
    }
    expectSym("(");
    do {
      std::string cname;
      if (acceptKw("CONSTRAINT") && !isKw("PRIMARY") && !isKw("UNIQUE") && !isKw("FOREIGN") && !isKw("CHECK")) cname = ident();
      if (acceptKw("FOREIGN")) { s.fks.push_back(fkDef(cname)); continue; }
      if (!cname.empty() && acceptKw("UNIQUE")) {  // CONSTRAINT name UNIQUE [KEY] (cols)
        if (!acceptKw("KEY")) acceptKw("INDEX");
        IndexDef ix;
        ix.unique = true;
        ix.name = cname;
        if (cur().k == Tok::ID || cur().k == Tok::QID) ix.name = ident();
        ix.cols = parenIdents();
        s.indexes.push_back(ix);
        continue;
      }
      if (acceptKw("PRIMARY")) { expectKw("KEY"); s.pkCols = parenIdents(); continue; }
      if (acceptKw("SORT")) { expectKw("KEY"); s.sortKey = parenIdent(); continue; }
      if (acceptKw("SHARD")) { acceptKw("KEY"); s.shardKey = parenIdents(true); s.hasShardKey = true; continue; }
      if (acceptKw("UNIQUE")) {
        if (!acceptKw("KEY")) acceptKw("INDEX");
        indexDef(s, true);
        continue;
      }
      if (acceptKw("KEY") || acceptKw("INDEX")) {
        indexDef(s, false);
        continue;
      }
      if (isKw("VECTOR") && (strcasecmp(t[p + 1].v.c_str(), "INDEX") == 0 || strcasecmp(t[p + 1].v.c_str(), "KEY") == 0)) {
        p += 2;
        indexDef(s, false);
        s.indexes.back().vector = true;
        continue;
      }
      if (acceptKw("FULLTEXT")) {  // FULLTEXT [USING VERSION n] [KEY | INDEX] [name] (cols)
        if (acceptKw("USING")) { acceptKw("VERSION"); if (cur().k == Tok::NUM) ++p; }
        if (!acceptKw("KEY")) acceptKw("INDEX");
        if (cur().k == Tok::ID || cur().k == Tok::QID) ident();
        for (auto& c : parenIdents()) s.fulltextCols.push_back(c);
        continue;
      }
      if (acceptKw("SPATIAL") || acceptKw("CHECK")) {
        while (!isSym(",") && !isSym(")") && cur().k != Tok::END) {
          if (acceptSym("(")) { int depth = 1; while (depth && cur().k != Tok::END) { if (isSym("(")) ++depth; if (isSym(")")) --depth; ++p; } } else ++p;
        }
        continue;
      }
      bool pk = false;
      ColumnDef c = columnDef(&pk);
      if (pk) s.pkCols = {c.name};
      if (c.ref) { c.ref->cols = {c.name}; s.fks.push_back(*c.ref); }
      if (c.unique) s.indexes.push_back(IndexDef{"", {c.name}, true});
      s.cols.push_back(c);
    } while (acceptSym(","));
    expectSym(")");
    while (cur().k != Tok::END && !isSym(";") && !isKw("AS") && !isKw("SELECT")) {  // table options
      if (acceptKw("VRAM")) s.vram = vramHint();
      else ++p;
    }
    if (acceptKw("AS") || isKw("SELECT")) s.select = std::make_shared<Stmt>(query());
    return s;
  }

  Stmt parse() {
    Stmt s;
    if (cur().k == Tok::END) return s;
    if (isKw("SELECT") || isKw("WITH") || isSym("(")) s = query();
    else if (isKw("EXPLAIN") && !(t[p + 1].k == Tok::ID && (strcasecmp(t[p + 1].v.c_str(), "SELECT") == 0 ||
                                                            strcasecmp(t[p + 1].v.c_str(), "WITH") == 0 ||
                                                            strcasecmp(t[p + 1].v.c_str(), "EXTENDED") == 0)) &&
             !(t[p + 1].k == Tok::SYM && t[p + 1].v == "(")) {
      ++p;  // EXPLAIN <table> = DESCRIBE
      s.type = Stmt::DESCRIBE;
      s.table = tableName();
    } else if (acceptKw("EXPLAIN")) {
      s.type = Stmt::EXPLAIN;
      acceptKw("EXTENDED");
      s.inner = std::make_shared<Stmt>(query());
    } else if (acceptKw("PROFILE")) {
      s.type = Stmt::PROFILE;
      s.inner = std::make_shared<Stmt>(query());
    } else if (acceptKw("CREATE")) {
      acceptKw("OR") && acceptKw("REPLACE");
      if (acceptKw("DATABASE") || acceptKw("SCHEMA")) {
        s.type = Stmt::CREATE_DB;
        if (acceptKw("IF")) { expectKw("NOT"); expectKw("EXISTS"); s.ifNotExists = true; }
        s.table = ident();
        while (cur().k != Tok::END && !isSym(";")) {  // [WITH] {SYNC | ASYNC} DURABILITY, CHARACTER SET .., ..
          if ((isKw("SYNC") || isKw("ASYNC")) && t[p + 1].k == Tok::ID && upper(t[p + 1].v) == "DURABILITY") s.setValue = upper(t[p].v);
          ++p;
        }
      } else if (isKw("VIEW") || (isKw("ALGORITHM") || isKw("DEFINER") || isKw("SQL"))) {
        while (!acceptKw("VIEW")) ++p;
        s.type = Stmt::CREATE_VIEW;
        if (acceptKw("IF")) { expectKw("NOT"); expectKw("EXISTS"); s.ifNotExists = true; }
        s.table = tableName();
        if (isSym("(")) s.insertCols = parenIdents();
        expectKw("AS");
        size_t start = p;
        s.select = std::make_shared<Stmt>(query());
        s.viewSql = joinTokens(start, p);
      } else if (isKw("INDEX") || ((isKw("UNIQUE") || isKw("VECTOR")) && strcasecmp(t[p + 1].v.c_str(), "INDEX") == 0)) {
        bool uq = acceptKw("UNIQUE");
        bool vec = acceptKw("VECTOR");
        expectKw("INDEX");
        s.type = Stmt::ALTER;
        Stmt::Alter a;
        a.kind = Stmt::Alter::ADD_INDEX;
        a.index.unique = uq;
        a.index.vector = vec;
        a.index.name = ident();
        if (acceptKw("USING")) ident();
        expectKw("ON");
        s.table = tableName();
        a.index.cols = parenIdents();
        if (acceptKw("INDEX_OPTIONS") && cur().k == Tok::STR) a.index.options = t[p++].v;
        s.alters.push_back(a);
        while (cur().k != Tok::END && !isSym(";")) ++p;
      } else if (acceptKw("PIPELINE")) {
        s.type = Stmt::PIPELINE;
        s.setName = "CREATE";
        if (acceptKw("IF")) { expectKw("NOT"); expectKw("EXISTS"); s.ifNotExists = true; }
        s.table = ident();
        expectKw("AS");
        s.inner = std::make_shared<Stmt>(loadData());
      } else if (acceptKw("USER")) {
        s.type = Stmt::USER;
        s.setName = "CREATE";
        if (acceptKw("IF")) { expectKw("NOT"); expectKw("EXISTS"); s.ifNotExists = true; }
        s.table = userName();
        s.path = identifiedBy();
        while (cur().k != Tok::END && !isSym(";")) ++p;
      } else if (acceptKw("ROLE")) {  // a role: a user that can't log in, granted to users
        s.type = Stmt::USER;
        s.setName = "CREATE";
        if (acceptKw("IF")) { expectKw("NOT"); expectKw("EXISTS"); s.ifNotExists = true; }
        s.table = userName();
        s.path = "\x03role";
      } else if (acceptKw("GROUP") ||
                 acceptKw("PROCEDURE") || acceptKw("FUNCTION") || acceptKw("TRIGGER") || acceptKw("RESOURCE")) {
        s.type = Stmt::SET;  // accepted and ignored
        while (cur().k != Tok::END) ++p;
      } else {
        bool ref = acceptKw("REFERENCE");
        if (!ref) acceptKw("ROWSTORE") || acceptKw("COLUMNSTORE");
        s = create();
        s.reference = ref;
      }
    } else if (acceptKw("DROP")) {
      s.type = Stmt::DROP;
      if (acceptKw("ALL")) {  // DROP ALL FROM PLANCACHE
        expectKw("FROM");
        expectKw("PLANCACHE");
        s.type = Stmt::DROP_PLANCACHE;
        acceptSym(";");
        return s;
      }
      if (acceptKw("DATABASE") || acceptKw("SCHEMA")) {
        s.type = Stmt::DROP_DB;
        if (acceptKw("IF")) { expectKw("EXISTS"); s.ifExists = true; }
        s.table = ident();
        return s;
      }
      if (acceptKw("VIEW")) {
        s.type = Stmt::DROP_VIEW;
        if (acceptKw("IF")) { expectKw("EXISTS"); s.ifExists = true; }
        do s.names.push_back(tableName()); while (acceptSym(","));
        return s;
      }
      if (acceptKw("INDEX")) {  // DROP INDEX name ON t
        s.type = Stmt::ALTER;
        Stmt::Alter a;
        a.kind = Stmt::Alter::DROP_INDEX;
        a.name = ident();
        expectKw("ON");
        s.table = tableName();
        s.alters.push_back(a);
        return s;
      }
      if (acceptKw("PIPELINE")) {
        s.type = Stmt::PIPELINE;
        s.setName = "DROP";
        if (acceptKw("IF")) { expectKw("EXISTS"); s.ifExists = true; }
        s.table = ident();
        return s;
      }
      if (acceptKw("USER")) {
        s.type = Stmt::USER;
        s.setName = "DROP";
        if (acceptKw("IF")) { expectKw("EXISTS"); s.ifExists = true; }
        s.table = userName();
        return s;
      }
      if (acceptKw("ROLE")) {
        s.type = Stmt::USER;
        s.setName = "DROP";
        if (acceptKw("IF")) { expectKw("EXISTS"); s.ifExists = true; }
        s.table = userName();
        return s;
      }
      if (acceptKw("PROCEDURE") || acceptKw("FUNCTION") || acceptKw("TRIGGER")) {
        s.type = Stmt::SET;
        while (cur().k != Tok::END) ++p;
        return s;
      }
      acceptKw("TEMPORARY");
      expectKw("TABLE");
      if (acceptKw("IF")) { expectKw("EXISTS"); s.ifExists = true; }
      do s.names.push_back(tableName()); while (acceptSym(","));
      s.table = s.names[0];
    } else if (isKw("ALTER") && strcasecmp(t[p + 1].v.c_str(), "USER") == 0) {
      p += 2;
      s.type = Stmt::USER;
      s.setName = "ALTER";
      if (acceptKw("IF")) expectKw("EXISTS");
      s.table = userName();
      s.path = identifiedBy();
      while (cur().k != Tok::END && !isSym(";")) ++p;
    } else if (isKw("SET") && strcasecmp(t[p + 1].v.c_str(), "PASSWORD") == 0) {  // SET PASSWORD [FOR u] = 'pw'
      p += 2;
      s.type = Stmt::USER;
      s.setName = "ALTER";
      s.table = acceptKw("FOR") ? userName() : "";
      expectSym("=");
      if (acceptKw("PASSWORD")) { expectSym("("); s.path = t[p++].v; expectSym(")"); }
      else s.path = t[p++].v;
    } else if (acceptKw("BACKUP") || acceptKw("RESTORE")) {
      // BACKUP DATABASE db [WITH ..] TO {'dir' | S3 'bucket/prefix' [CONFIG '{..}'] [CREDENTIALS '{..}']}
      // RESTORE DATABASE db [AS new] FROM {'dir' | S3 ..} [UNTIL TIMESTAMP 'yyyy-mm-dd hh:mm:ss']
      s.type = Stmt::BACKUP;
      s.setName = upper(t[p - 1].v);
      if (!acceptKw("DATABASE")) expectKw("SCHEMA");
      s.table = ident();
      if (acceptKw("AS")) s.asName = ident();
      while (cur().k == Tok::ID && !isKw("TO") && !isKw("FROM")) ++p;  // WITH INIT / DIFFERENTIAL ..
      if (!acceptKw("TO")) expectKw("FROM");
      if (acceptKw("S3")) s.setValue = "S3";
      if (cur().k != Tok::STR) fail("path");
      s.path = t[p++].v;
      for (;;) {
        if (acceptKw("CONFIG") && cur().k == Tok::STR) s.config = t[p++].v;
        else if (acceptKw("CREDENTIALS") && cur().k == Tok::STR) s.credentials = t[p++].v;
        else if (acceptKw("UNTIL")) {
          acceptKw("TIMESTAMP");
          if (cur().k != Tok::STR) fail("timestamp");
          s.until = t[p++].v;
        } else break;
      }
      while (cur().k != Tok::END && !isSym(";")) ++p;
    } else if (acceptKw("ALTER")) {
      if (!acceptKw("TABLE")) {  // ALTER DATABASE / USER / ..: ignored
        s.type = Stmt::SET;
        while (cur().k != Tok::END) ++p;
        return s;
      }
      s.type = Stmt::ALTER;
      s.table = tableName();
      do {
        Stmt::Alter a;
        if (acceptKw("ADD")) {
          std::string cname;
          bool constraint = acceptKw("CONSTRAINT");
          if (constraint && !isKw("PRIMARY") && !isKw("UNIQUE") && !isKw("FOREIGN") && !isKw("CHECK")) cname = ident();
          if (acceptKw("FOREIGN")) { a.kind = Stmt::Alter::ADD_FK; a.fk = fkDef(cname); }
          else if (constraint && acceptKw("CHECK")) { a.kind = Stmt::Alter::NOOP; while (cur().k != Tok::END && !isSym(",")) ++p; }
          else if (constraint && acceptKw("UNIQUE")) {
            if (!acceptKw("KEY")) acceptKw("INDEX");
            a.kind = Stmt::Alter::ADD_INDEX;
            a.index.unique = true;
            a.index.name = cname;
            if (cur().k == Tok::ID || cur().k == Tok::QID) a.index.name = ident();
            a.index.cols = parenIdents();
          }
          else if (acceptKw("PRIMARY")) { expectKw("KEY"); parenIdents(); a.kind = Stmt::Alter::NOOP; }
          else if (acceptKw("UNIQUE") || isKw("KEY") || isKw("INDEX") || isKw("VECTOR")) {
            bool uq = t[p - 1].v.size() && strcasecmp(t[p - 1].v.c_str(), "UNIQUE") == 0;
            a.index.vector = acceptKw("VECTOR");
            if (!acceptKw("KEY")) acceptKw("INDEX");
            a.kind = Stmt::Alter::ADD_INDEX;
            a.index.unique = uq;
            if (cur().k == Tok::ID || cur().k == Tok::QID) a.index.name = ident();
            a.index.cols = parenIdents();
            if (acceptKw("INDEX_OPTIONS") && cur().k == Tok::STR) a.index.options = t[p++].v;
          } else {
            acceptKw("COLUMN");
            a.kind = Stmt::Alter::ADD_COL;
            a.col = columnDef();
            if (acceptKw("FIRST")) a.newName = "\x01FIRST";
            else if (acceptKw("AFTER")) a.newName = ident();
          }
        } else if (acceptKw("DROP")) {
          if (acceptKw("PRIMARY")) { expectKw("KEY"); a.kind = Stmt::Alter::NOOP; }
          else if (acceptKw("FOREIGN")) { expectKw("KEY"); a.kind = Stmt::Alter::DROP_FK; a.name = ident(); }
          else if (acceptKw("CONSTRAINT") || acceptKw("CHECK")) { a.kind = Stmt::Alter::DROP_FK; a.name = ident(); }  // FK, unique key or check
          else if (acceptKw("INDEX") || acceptKw("KEY")) { a.kind = Stmt::Alter::DROP_INDEX; a.name = ident(); }
          else { acceptKw("COLUMN"); a.kind = Stmt::Alter::DROP_COL; a.name = ident(); }
        } else if (acceptKw("RENAME")) {
          if (acceptKw("COLUMN")) {
            a.kind = Stmt::Alter::RENAME_COL;
            a.name = ident();
            expectKw("TO");
            a.newName = ident();
          } else {
            if (!acceptKw("TO")) acceptKw("AS");
            a.kind = Stmt::Alter::RENAME_TABLE;
            a.newName = tableName();
          }
        } else if (acceptKw("MODIFY")) {
          acceptKw("COLUMN");
          a.kind = Stmt::Alter::MODIFY_COL;
          a.col = columnDef();
          a.name = a.col.name;
        } else if (acceptKw("CHANGE")) {
          acceptKw("COLUMN");
          a.kind = Stmt::Alter::MODIFY_COL;
          a.name = ident();
          a.col = columnDef();
        } else if (acceptKw("VRAM")) {  // ALTER TABLE t VRAM = PIN
          a.kind = Stmt::Alter::SET_VRAM;
          a.vram = vramHint();
        } else if (acceptKw("ALTER")) {  // ALTER [COLUMN] c SET VRAM = PIN
          acceptKw("COLUMN");
          a.name = ident();
          if (acceptKw("SET") && acceptKw("VRAM")) {
            a.kind = Stmt::Alter::SET_VRAM;
            a.vram = vramHint();
          } else {
            while (cur().k != Tok::END && !isSym(",")) ++p;  // SET DEFAULT .. / DROP DEFAULT
          }
        } else {
          while (cur().k != Tok::END && !isSym(",")) ++p;  // table options, ALGORITHM=.., etc.
        }
        s.alters.push_back(a);
      } while (acceptSym(","));
    } else if (acceptKw("SNAPSHOT")) {  // SNAPSHOT [DATABASE [db]]: checkpoint, truncates the WAL
      s.type = Stmt::SNAPSHOT;
      while (cur().k != Tok::END && !isSym(";")) ++p;
    } else if (acceptKw("TRUNCATE")) {
      s.type = Stmt::TRUNCATE;
      acceptKw("TABLE");
      s.table = tableName();
    } else if (acceptKw("RENAME")) {
      s.type = Stmt::RENAME;
      expectKw("TABLE");
      do {
        s.names.push_back(tableName());
        expectKw("TO");
        s.names.push_back(tableName());
      } while (acceptSym(","));
    } else if (isKw("INSERT") || isKw("REPLACE")) {
      s.type = Stmt::INSERT;
      s.replaceInto = acceptKw("REPLACE");
      if (!s.replaceInto) expectKw("INSERT");
      if (acceptKw("IGNORE")) s.insertIgnore = true;
      acceptKw("INTO");
      s.table = tableName();
      if (acceptSym("(")) {
        if (isKw("SELECT") || isKw("WITH")) {
          --p;  // INSERT INTO t (SELECT ..)
        } else {
          do s.insertCols.push_back(ident()); while (acceptSym(","));
          expectSym(")");
        }
      }
      if (isKw("SELECT") || isKw("WITH") || isSym("(")) {
        s.select = std::make_shared<Stmt>(query());
      } else if (acceptKw("SET")) {  // INSERT INTO t SET a = 1, b = 2
        std::vector<ExprP> row;
        do {
          s.insertCols.push_back(ident());
          expectSym("=");
          row.push_back(expr());
        } while (acceptSym(","));
        s.rows.push_back(std::move(row));
      } else {
        if (!acceptKw("VALUES")) expectKw("VALUE");
        do {
          expectSym("(");
          std::vector<ExprP> row;
          if (!isSym(")")) do row.push_back(expr()); while (acceptSym(","));
          expectSym(")");
          s.rows.push_back(std::move(row));
        } while (acceptSym(","));
      }
      if (acceptKw("ON")) {
        expectKw("DUPLICATE");
        expectKw("KEY");
        expectKw("UPDATE");
        do {
          std::string c = ident();
          if (acceptSym(".")) c = ident();
          expectSym("=");
          s.onDup.push_back({c, expr()});
        } while (acceptSym(","));
      }
    } else if (acceptKw("DELETE") && !isKw("FROM")) {  // DELETE alias FROM t alias JOIN u .. WHERE ..
      s.type = Stmt::DELETE;
      s.targetAlias = ident();
      expectKw("FROM");
      tableRefs(s);
      for (auto& f : s.from)
        if (strcasecmp(f.alias.c_str(), s.targetAlias.c_str()) == 0) s.table = f.name;
      if (s.table.empty()) throw SqlError("Unknown table '" + s.targetAlias + "' in MULTI DELETE", 1109);
      if (acceptKw("WHERE")) s.where = expr();
    } else if (p > 0 && upper(t[p - 1].v) == "DELETE") {  // DELETE FROM t [alias] [JOIN ..] WHERE .. [ORDER BY ..] [LIMIT n]
      s.type = Stmt::DELETE;
      expectKw("FROM");
      tableRefs(s);
      s.table = s.from[0].name;
      s.targetAlias = s.from[0].alias;
      if (acceptKw("WHERE")) s.where = expr();
      if (acceptKw("ORDER")) {
        expectKw("BY");
        do {
          OrderItem o{expr(), false};
          if (acceptKw("DESC")) o.desc = true;
          else acceptKw("ASC");
          s.orderBy.push_back(o);
        } while (acceptSym(","));
      }
      if (acceptKw("LIMIT")) s.limit = intLit();
    } else if (acceptKw("UPDATE")) {  // UPDATE t [alias] [JOIN ..] SET [a.]c = .. WHERE .. [ORDER BY ..] [LIMIT n]
      s.type = Stmt::UPDATE;
      tableRefs(s);
      s.table = s.from[0].name;
      s.targetAlias = s.from[0].alias;
      expectKw("SET");
      bool first = true;
      do {
        std::string c = ident();
        if (acceptSym(".")) {
          if (first) s.targetAlias = c;
          c = ident();
        }
        first = false;
        expectSym("=");
        s.sets.push_back({c, expr()});
      } while (acceptSym(","));
      for (auto& f : s.from)
        if (strcasecmp(f.alias.c_str(), s.targetAlias.c_str()) == 0) s.table = f.name;
      if (acceptKw("WHERE")) s.where = expr();
      if (acceptKw("ORDER")) {
        expectKw("BY");
        do {
          OrderItem o{expr(), false};
          if (acceptKw("DESC")) o.desc = true;
          else acceptKw("ASC");
          s.orderBy.push_back(o);
        } while (acceptSym(","));
      }
      if (acceptKw("LIMIT")) s.limit = intLit();
    } else if (isKw("LOAD")) {
      s = loadData();
    } else if (acceptKw("SHOW")) {
      bool full = acceptKw("FULL");
      auto likeOrFrom = [&]() {
        if (acceptKw("FROM") || acceptKw("IN")) s.table = tableName();
        if (acceptKw("FROM") || acceptKw("IN")) s.table = ident() + "." + s.table;
        if (acceptKw("LIKE") && cur().k == Tok::STR) s.path = t[p++].v;
        if (acceptKw("WHERE")) s.where = expr();
      };
      if (acceptKw("TABLES")) {
        s.type = Stmt::SHOW_TABLES;
        s.setName = full ? "FULL" : "";
        if (acceptKw("FROM") || acceptKw("IN")) s.table = ident();
        if (acceptKw("LIKE") && cur().k == Tok::STR) s.path = t[p++].v;
      } else if (acceptKw("DATABASES") || acceptKw("SCHEMAS")) {
        s.type = Stmt::SHOW_DATABASES;
        if (acceptKw("LIKE") && cur().k == Tok::STR) s.path = t[p++].v;
      } else if (acceptKw("CREATE")) {
        s.type = Stmt::SHOW_MISC;
        if (acceptKw("VIEW")) s.setName = "CREATE VIEW";
        else if (acceptKw("DATABASE") || acceptKw("SCHEMA")) s.setName = "CREATE DATABASE";
        else { expectKw("TABLE"); s.setName = "CREATE TABLE"; }
        s.table = tableName();
      } else if (acceptKw("INDEX") || acceptKw("INDEXES") || acceptKw("KEYS")) {
        s.type = Stmt::SHOW_MISC;
        s.setName = "INDEX";
        likeOrFrom();
      } else if (acceptKw("VARIABLES") || acceptKw("STATUS")) {
        s.type = Stmt::SHOW_MISC;
        s.setName = upper(t[p - 1].v);
        likeOrFrom();
      } else if (isKw("GLOBAL") || isKw("SESSION")) {
        ++p;
        s.type = Stmt::SHOW_MISC;
        s.setName = upper(ident());
        likeOrFrom();
      } else if (acceptKw("TABLE")) {
        expectKw("STATUS");
        s.type = Stmt::SHOW_MISC;
        s.setName = "TABLE STATUS";
        likeOrFrom();
      } else if (acceptKw("SEGMENTS")) {
        s.type = Stmt::SHOW_SEGMENTS;
        if (!acceptKw("FROM")) expectKw("IN");
        s.table = tableName();
      } else if (acceptKw("PROFILE")) {
        s.type = Stmt::SHOW_PROFILE;
      } else if (acceptKw("PLANCACHE")) {
        s.type = Stmt::SHOW_PLANCACHE;
      } else if (acceptKw("PLAN")) {
        s.type = Stmt::SHOW_PLAN;
        s.planId = intLit();
      } else if (acceptKw("COLUMNS") || acceptKw("FIELDS")) {
        s.type = Stmt::DESCRIBE;
        if (!acceptKw("FROM")) expectKw("IN");
        s.table = tableName();
        if (acceptKw("FROM") || acceptKw("IN")) s.table = ident() + "." + s.table;
      } else if (cur().k == Tok::ID) {
        s.type = Stmt::SHOW_MISC;  // WARNINGS, ERRORS, ENGINES, COLLATION, CHARACTER SET, PROCESSLIST, GRANTS, ..
        s.setName = upper(ident());
        if (full) s.setValue = "FULL";
        if (s.setName == "CHARACTER") { acceptKw("SET"); s.setName = "CHARSET"; }
        likeOrFrom();
      } else {
        s.type = Stmt::SET;
      }
      while (cur().k != Tok::END) ++p;
      return s;
    } else if (acceptKw("DESCRIBE") || acceptKw("DESC") || acceptKw("EXPLAIN")) {
      s.type = Stmt::DESCRIBE;
      s.table = tableName();
    } else if (acceptKw("OPTIMIZE") || acceptKw("ANALYZE")) {
      s.type = Stmt::OPTIMIZE;
      expectKw("TABLE");
      s.table = tableName();
      while (cur().k != Tok::END && !isSym(";")) ++p;
    } else if (acceptKw("USE")) {
      s.type = Stmt::USE;
      s.table = ident();
    } else if ((isKw("START") || isKw("STOP")) && strcasecmp(t[p + 1].v.c_str(), "PIPELINE") == 0) {
      s.type = Stmt::PIPELINE;
      s.setName = upper(t[p].v);
      p += 2;
      s.table = ident();
      while (cur().k != Tok::END && !isSym(";")) ++p;  // FOREGROUND, LIMIT n BATCHES
    } else if (isKw("START") && strcasecmp(t[p + 1].v.c_str(), "TRANSACTION") == 0) {
      s.type = Stmt::TXN;
      s.setName = "BEGIN";
      while (cur().k != Tok::END) ++p;
    } else if (isKw("GRANT") || isKw("REVOKE")) {
      // GRANT privs ON scope TO users [WITH GRANT OPTION] | GRANT role TO users; REVOKE .. FROM ..
      s.type = Stmt::USER;
      s.setName = upper(t[p++].v);
      for (;;) {
        std::string pv;
        while (cur().k != Tok::END && !isSym(",") && !isKw("ON") && !isKw("TO") && !isKw("FROM")) {
          if (isSym("(")) {  // column list: privilege applies to the whole table here
            while (cur().k != Tok::END && !isSym(")")) ++p;
            ++p;
            continue;
          }
          pv += (pv.empty() ? "" : " ") + upper(t[p++].v);
        }
        if (pv == "ALL PRIVILEGES") pv = "ALL";
        s.names.push_back(pv);
        if (!acceptSym(",")) break;
      }
      if (acceptKw("ON")) {
        acceptKw("TABLE");
        std::string sc;
        while (cur().k != Tok::END && !isKw("TO") && !isKw("FROM")) sc += t[p++].v;
        if (sc == "*") sc = "*.*";
        for (auto& ch : sc) ch = (char)std::tolower((unsigned char)ch);
        s.path = sc;
      } else {
        s.setName += "_ROLE";
      }
      if (!acceptKw("TO")) expectKw("FROM");
      do s.grantees.push_back(userName()); while (acceptSym(","));
      while (cur().k != Tok::END && !isSym(";")) ++p;
    } else if (acceptKw("FLUSH") || acceptKw("KILL") || acceptKw("LOCK") ||
               acceptKw("UNLOCK") || acceptKw("START") || acceptKw("STOP") || acceptKw("RESET")) {
      s.type = Stmt::SET;  // accepted and ignored
      while (cur().k != Tok::END) ++p;
    } else if (acceptKw("SET")) {
      s.type = Stmt::SET;
      if (!acceptKw("SESSION")) acceptKw("GLOBAL");
      if (cur().k == Tok::ID) {
        s.setName = upper(t[p++].v);
        if (s.setName.rfind("@@", 0) == 0) {  // SET @@[session.|global.|local.]name = v
          s.setName = s.setName.substr(2);
          if ((s.setName == "SESSION" || s.setName == "GLOBAL" || s.setName == "LOCAL") && acceptSym(".")) s.setName = upper(ident());
        }
        if ((acceptSym("=") || acceptSym(":=")) && cur().k != Tok::END) s.setValue = upper(t[p++].v);
      }
      while (cur().k != Tok::END) ++p;
    } else if (acceptKw("SAVEPOINT")) {
      s.type = Stmt::TXN;
      s.setName = "SAVEPOINT " + ident();
    } else if (isKw("RELEASE") && upper(t[p + 1].v) == "SAVEPOINT") {
      p += 2;
      s.type = Stmt::TXN;
      s.setName = "RELEASE " + ident();
    } else if (isKw("ROLLBACK") && upper(t[p + 1].v) == "TO") {
      p += 2;
      acceptKw("SAVEPOINT");
      s.type = Stmt::TXN;
      s.setName = "ROLLBACK TO " + ident();
    } else if (acceptKw("BEGIN") || acceptKw("COMMIT") || acceptKw("ROLLBACK") || acceptKw("START")) {
      s.type = Stmt::TXN;
      s.setName = upper(t[p - 1].v) == "START" ? "BEGIN" : upper(t[p - 1].v);
      while (cur().k != Tok::END) ++p;
    } else {
      fail("statement");
    }
    acceptSym(";");
    if (cur().k != Tok::END) fail("end of statement");
    return s;
  }
};

}  // namespace

ExprP (*udfHook)(const ExprP&) = nullptr;

Stmt parseSql(const std::string& sql) {
  Parser ps;
  ps.t = lex(sql);
  return ps.parse();
}

}  // namespace gdb
