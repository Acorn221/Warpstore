// Stored procedures and user-defined functions (SingleStore MPSQL / MySQL-style subset).
//
//   CREATE [OR REPLACE] PROCEDURE p(a INT, b TEXT) [RETURNS type] AS
//   DECLARE v INT = 0; ..
//   BEGIN
//     v = v + a;  /  SET v = ..;  /  SELECT x INTO v FROM ..;  /  v = SCALAR(SELECT ..);
//     IF c THEN .. ELSEIF c THEN .. ELSE .. END IF;
//     WHILE c LOOP .. END LOOP;  /  WHILE c DO .. END WHILE;  /  LOOP .. EXIT WHEN c; END LOOP;
//     FOR i IN 1 .. 10 LOOP .. END LOOP;  /  FOR r IN SELECT .. LOOP .. r.col .. END LOOP;
//     ECHO SELECT ..;  (or a bare SELECT: becomes the CALL's result)
//     RAISE USER_EXCEPTION('msg');  /  SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'msg';
//     BEGIN .. EXCEPTION WHEN OTHERS THEN .. END;
//     RETURN expr;  any other SQL statement runs with variables substituted as literals
//   END
//   CALL p(1, 'x');  ECHO p(..);  DROP PROCEDURE [IF EXISTS] p;  SHOW PROCEDURES / FUNCTIONS
//
// Expressions are evaluated by running "SELECT <expr>" through the engine. A function whose body is
// a single RETURN expr is inlined into queries at parse time (so it runs on the GPU); other functions
// can be called with constant arguments.
#include <strings.h>

#include <cctype>
#include <functional>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>

#include "engine.h"
#include "sql.h"
#include "storage.h"

namespace gdb {

extern thread_local bool tApplying;

namespace {

struct PTok {
  enum K { WORD, STR, NUM, SYM, QID } k;
  std::string v;      // WORD: as written; STR: unquoted
  size_t b = 0, e = 0;  // span in the source
};

std::vector<PTok> lex(const std::string& s) {
  std::vector<PTok> out;
  size_t i = 0;
  while (i < s.size()) {
    char c = s[i];
    if (isspace((unsigned char)c)) { ++i; continue; }
    if (c == '-' && i + 1 < s.size() && s[i + 1] == '-') { while (i < s.size() && s[i] != '\n') ++i; continue; }
    if (c == '#') { while (i < s.size() && s[i] != '\n') ++i; continue; }
    if (c == '/' && i + 1 < s.size() && s[i + 1] == '*') {
      size_t e = s.find("*/", i + 2);
      i = e == std::string::npos ? s.size() : e + 2;
      continue;
    }
    PTok t;
    t.b = i;
    if (c == '\'' || c == '"' || c == '`') {
      char q = c;
      std::string v;
      ++i;
      while (i < s.size()) {
        if (s[i] == '\\' && i + 1 < s.size()) { v += s[i + 1]; i += 2; continue; }
        if (s[i] == q) {
          if (i + 1 < s.size() && s[i + 1] == q) { v += q; i += 2; continue; }
          ++i;
          break;
        }
        v += s[i++];
      }
      t.k = q == '`' ? PTok::QID : PTok::STR;
      t.v = v;
    } else if (isalpha((unsigned char)c) || c == '_' || c == '@' || c == '$') {
      while (i < s.size() && (isalnum((unsigned char)s[i]) || s[i] == '_' || s[i] == '@' || s[i] == '$')) ++i;
      t.k = PTok::WORD;
      t.v = s.substr(t.b, i - t.b);
    } else if (isdigit((unsigned char)c)) {
      while (i < s.size() && (isalnum((unsigned char)s[i]) || (s[i] == '.' && !(i + 1 < s.size() && s[i + 1] == '.')))) ++i;
      t.k = PTok::NUM;
      t.v = s.substr(t.b, i - t.b);
    } else {
      static const char* two[] = {":=", "..", "<=", ">=", "<>", "!=", "||", "&&", "->", "<*", "*>", "<-"};
      t.k = PTok::SYM;
      t.v = std::string(1, c);
      for (auto* x : two)
        if (s.compare(i, 2, x) == 0) t.v = x;
      i += t.v.size();
    }
    t.e = i;
    out.push_back(t);
  }
  return out;
}

bool kw(const PTok& t, const char* w) { return t.k == PTok::WORD && strcasecmp(t.v.c_str(), w) == 0; }
bool sym(const PTok& t, const char* w) { return t.k == PTok::SYM && t.v == w; }

struct Node {
  enum K { SQL, ASSIGN, IF, WHILE, LOOP, FOR_RANGE, FOR_QUERY, EXIT, CONTINUE, RETURN, ECHO, RAISE, BLOCK, DECLARE } k;
  size_t b = 0, e = 0;          // token range of the SQL / expression
  std::string var;              // ASSIGN / FOR / DECLARE target(s)
  std::vector<std::pair<size_t, size_t>> conds;  // IF: conditions; WHILE: one; FOR_RANGE: lo, hi; EXIT WHEN
  std::vector<std::vector<Node>> bodies;         // IF: one per cond (+ else); loops: one; BLOCK: body, handler
  bool hasElse = false;
  std::vector<std::string> intoVars;  // SELECT .. INTO a, b
};

struct Routine {
  std::string name, createSql;
  bool isFunc = false;
  std::vector<std::string> params;
  std::string src;  // the body source (tokens index into it)
  std::vector<PTok> toks;
  std::vector<Node> decls, body;
  std::string inlineExpr;  // functions with a single RETURN expr
  std::string retType = "VARCHAR";  // declared RETURNS type, as BIGINT / DOUBLE / DECIMAL / DATE / DATETIME / VARCHAR
};

struct Parser {
  const std::vector<PTok>& t;
  size_t p = 0;
  explicit Parser(const std::vector<PTok>& t) : t(t) {}
  bool end() const { return p >= t.size(); }
  [[noreturn]] void fail(const std::string& what) {
    throw SqlError("procedure syntax error near '" + (end() ? std::string("end") : t[p].v) + "': expected " + what, 1064);
  }
  bool atKw(const char* w) const { return !end() && kw(t[p], w); }
  bool atKw2(const char* a, const char* b) const { return p + 1 < t.size() && kw(t[p], a) && kw(t[p + 1], b); }
  bool acc(const char* w) { if (atKw(w)) { ++p; return true; } return false; }
  void expect(const char* w) { if (!acc(w)) fail(w); }
  // Tokens up to the next top-level ';' (or one of the stop words at depth 0).
  size_t until(std::initializer_list<const char*> stops) {
    int depth = 0;
    while (!end()) {
      if (sym(t[p], "(")) ++depth;
      else if (sym(t[p], ")")) --depth;
      else if (depth == 0 && sym(t[p], ";")) break;
      else if (depth == 0)
        for (auto* s : stops)
          if (kw(t[p], s)) return p;
      ++p;
    }
    return p;
  }
  void semi() {
    if (!end() && sym(t[p], ";")) ++p;
  }
  bool atEnd() const {  // END / ELSE / ELSEIF / EXCEPTION / UNTIL close a statement list
    return end() || atKw("END") || atKw("ELSE") || atKw("ELSEIF") || atKw("EXCEPTION");
  }
  std::vector<Node> list() {
    std::vector<Node> out;
    while (!atEnd()) {
      if (sym(t[p], ";")) { ++p; continue; }
      out.push_back(stmt());
    }
    return out;
  }
  void endOf(const char* what) {  // END [IF|LOOP|WHILE]
    expect("END");
    if (what) acc(what);
    semi();
  }
  Node stmt() {
    Node n;
    // label:  (MySQL loop labels) are skipped
    if (p + 1 < t.size() && t[p].k == PTok::WORD && sym(t[p + 1], ":") && !(p + 2 < t.size() && sym(t[p + 2], "="))) p += 2;
    if (acc("DECLARE")) {
      n.k = Node::DECLARE;
      n.var = t[p++].v;
      size_t b = p;
      (void)b;
      while (!end() && !sym(t[p], ";") && !sym(t[p], "=") && !atKw("DEFAULT") && !sym(t[p], ":=")) ++p;
      if (!end() && (sym(t[p], "=") || sym(t[p], ":=") || atKw("DEFAULT"))) {
        ++p;
        n.b = p;
        n.e = until({});
      }
      semi();
      return n;
    }
    if (acc("IF")) {
      n.k = Node::IF;
      for (;;) {
        size_t b = p;
        size_t e = until({"THEN"});
        expect("THEN");
        n.conds.push_back({b, e});
        n.bodies.push_back(list());
        if (acc("ELSEIF") || atKw2("ELSE", "IF")) {
          if (atKw("ELSE")) p += 2;
          continue;
        }
        if (acc("ELSE")) {
          n.bodies.push_back(list());
          n.hasElse = true;
        }
        break;
      }
      endOf("IF");
      return n;
    }
    if (acc("WHILE")) {
      n.k = Node::WHILE;
      size_t b = p, e = until({"LOOP", "DO"});
      n.conds.push_back({b, e});
      ++p;
      n.bodies.push_back(list());
      expect("END");
      if (!acc("LOOP")) acc("WHILE");
      if (!end() && t[p].k == PTok::WORD && !sym(t[p], ";")) ++p;  // label
      semi();
      return n;
    }
    if (acc("REPEAT")) {  // REPEAT .. UNTIL c END REPEAT: a LOOP with EXIT WHEN c at the end
      n.k = Node::LOOP;
      std::vector<Node> body;
      while (!end() && !atKw("UNTIL")) {
        if (sym(t[p], ";")) { ++p; continue; }
        body.push_back(stmt());
      }
      expect("UNTIL");
      Node ex;
      ex.k = Node::EXIT;
      size_t b = p, e = until({"END"});
      ex.conds.push_back({b, e});
      body.push_back(ex);
      n.bodies.push_back(body);
      endOf("REPEAT");
      return n;
    }
    if (acc("LOOP")) {
      n.k = Node::LOOP;
      n.bodies.push_back(list());
      endOf("LOOP");
      return n;
    }
    if (acc("FOR")) {
      n.var = t[p++].v;
      expect("IN");
      acc("REVERSE");
      size_t b = p;
      if (atKw("SELECT") || atKw("WITH") || (!end() && sym(t[p], "("))) {
        n.k = Node::FOR_QUERY;
        size_t e = until({"LOOP"});
        n.b = b;
        n.e = e;
      } else {
        n.k = Node::FOR_RANGE;
        int depth = 0;
        while (!end() && !(depth == 0 && sym(t[p], ".."))) {
          if (sym(t[p], "(")) ++depth;
          if (sym(t[p], ")")) --depth;
          ++p;
        }
        n.conds.push_back({b, p});
        ++p;
        size_t b2 = p, e2 = until({"LOOP"});
        n.conds.push_back({b2, e2});
      }
      expect("LOOP");
      n.bodies.push_back(list());
      endOf("LOOP");
      return n;
    }
    if (acc("EXIT") || acc("LEAVE") || acc("BREAK")) {
      n.k = Node::EXIT;
      if (!end() && t[p].k == PTok::WORD && !atKw("WHEN")) ++p;  // label
      if (acc("WHEN")) {
        size_t b = p, e = until({});
        n.conds.push_back({b, e});
      }
      semi();
      return n;
    }
    if (acc("CONTINUE") || acc("ITERATE")) {
      n.k = Node::CONTINUE;
      if (!end() && t[p].k == PTok::WORD && !atKw("WHEN")) ++p;
      if (acc("WHEN")) {
        size_t b = p, e = until({});
        n.conds.push_back({b, e});
      }
      semi();
      return n;
    }
    if (acc("RETURN")) {
      n.k = Node::RETURN;
      n.b = p;
      n.e = until({});
      semi();
      return n;
    }
    if (acc("ECHO")) {
      n.k = Node::ECHO;
      n.b = p;
      n.e = until({});
      semi();
      return n;
    }
    if (acc("RAISE")) {
      n.k = Node::RAISE;
      n.b = p;
      n.e = until({});
      semi();
      return n;
    }
    if (atKw("SIGNAL")) {  // SIGNAL SQLSTATE '45000' [SET MESSAGE_TEXT = expr]
      n.k = Node::RAISE;
      while (!end() && !sym(t[p], ";") && !atKw("MESSAGE_TEXT")) ++p;
      if (acc("MESSAGE_TEXT")) {
        ++p;  // =
        n.b = p;
        n.e = until({});
      } else {
        n.b = n.e = p;
      }
      semi();
      return n;
    }
    if (atKw("BEGIN") && !(p + 1 < t.size() && (kw(t[p + 1], "TRANSACTION") || kw(t[p + 1], "WORK") || sym(t[p + 1], ";")))) {
      ++p;
      n.k = Node::BLOCK;
      n.bodies.push_back(list());
      if (acc("EXCEPTION")) {  // EXCEPTION WHEN .. THEN ..
        while (!end() && !atKw("THEN")) ++p;
        expect("THEN");
        n.bodies.push_back(list());
      }
      endOf(nullptr);
      return n;
    }
    // Assignment: [SET] var = expr / var := expr.
    size_t save = p;
    bool set = acc("SET");
    if (!end() && (t[p].k == PTok::WORD) && p + 1 < t.size() && (sym(t[p + 1], "=") || sym(t[p + 1], ":="))) {
      n.k = Node::ASSIGN;
      n.var = t[p].v;
      p += 2;
      n.b = p;
      n.e = until({});
      semi();
      return n;
    }
    (void)set;
    p = save;
    n.k = Node::SQL;
    n.b = p;
    n.e = until({});
    semi();
    // SELECT .. INTO a, b [FROM ..]
    if (kw(t[n.b], "SELECT")) {
      int depth = 0;
      for (size_t i = n.b; i < n.e; ++i) {
        if (sym(t[i], "(")) ++depth;
        if (sym(t[i], ")")) --depth;
        if (depth == 0 && kw(t[i], "INTO") && i + 1 < n.e && !kw(t[i + 1], "OUTFILE") && !kw(t[i + 1], "DUMPFILE")) {
          size_t j = i + 1;
          std::string vars;
          for (;;) {
            n.intoVars.push_back(t[j].v);
            ++j;
            if (j < n.e && sym(t[j], ",")) { ++j; continue; }
            break;
          }
          n.var = std::to_string(i) + ":" + std::to_string(j);  // token span of the INTO clause
          break;
        }
      }
    }
    return n;
  }
};

thread_local std::map<std::string, Value> tUserVars;
std::mutex gRoutinesMu;
std::map<std::string, std::shared_ptr<Routine>> gRoutines;  // lower-case name ->
Engine* gEngine = nullptr;
thread_local int tCallDepth = 0;

struct Var {
  bool null = true, num = false;
  std::string s;
  std::string lit() const {
    if (null) return "NULL";
    if (num) return s;
    std::string o = "'";
    for (char c : s) {
      if (c == '\'' || c == '\\') o += '\\';
      o += c;
    }
    return o + "'";
  }
};

struct ExitLoop {};
struct ContinueLoop {};
struct ReturnNow {
  Var v;
};

std::shared_ptr<Routine> parseRoutine(const std::string& sql) {
  // CREATE [OR REPLACE] [DEFINER = ..] PROCEDURE|FUNCTION name ( params ) [RETURNS type] [attrs] [AS] body
  auto all = lex(sql);
  size_t p = 0;
  auto r = std::make_shared<Routine>();
  r->createSql = sql;
  while (p < all.size() && !kw(all[p], "PROCEDURE") && !kw(all[p], "FUNCTION")) ++p;
  if (p >= all.size()) throw SqlError("bad CREATE PROCEDURE", 1064);
  r->isFunc = kw(all[p], "FUNCTION");
  ++p;
  if (p + 2 < all.size() && kw(all[p], "IF")) p += 3;  // IF NOT EXISTS
  r->name = all[p++].v;
  if (p + 1 < all.size() && sym(all[p], ".")) r->name = all[++p].v, ++p;
  if (p >= all.size() || !sym(all[p], "(")) throw SqlError("expected ( after routine name", 1064);
  ++p;
  int depth = 0;
  bool expectName = true;
  while (p < all.size() && !(depth == 0 && sym(all[p], ")"))) {
    if (sym(all[p], "(")) ++depth;
    else if (sym(all[p], ")")) --depth;
    else if (depth == 0 && sym(all[p], ",")) expectName = true;
    else if (expectName && all[p].k == PTok::WORD && !kw(all[p], "IN") && !kw(all[p], "OUT") && !kw(all[p], "INOUT")) {
      r->params.push_back(lower(all[p].v));
      expectName = false;
    }
    ++p;
  }
  ++p;
  // Skip RETURNS type / characteristics up to AS / BEGIN / DECLARE / RETURN.
  if (p + 1 < all.size() && kw(all[p], "RETURNS")) {
    const std::string t = upper(all[p + 1].v);
    r->retType = t.find("INT") != std::string::npos || t == "BOOL" || t == "BOOLEAN" ? "BIGINT"
               : t == "DOUBLE" || t == "FLOAT" || t == "REAL" ? "DOUBLE" : t == "DECIMAL" || t == "NUMERIC" ? "DECIMAL"
               : t == "DATE" ? "DATE" : t == "DATETIME" || t == "TIMESTAMP" ? "DATETIME" : "VARCHAR";
  }
  while (p < all.size() && !kw(all[p], "AS") && !kw(all[p], "BEGIN") && !kw(all[p], "DECLARE") &&
         !(kw(all[p], "RETURN") && !(p > 0 && kw(all[p - 1], "RETURNS"))))
    ++p;
  if (p < all.size() && kw(all[p], "AS")) ++p;
  if (p >= all.size()) throw SqlError("routine has no body", 1064);
  r->src = sql;
  r->toks.assign(all.begin() + p, all.end());
  Parser ps(r->toks);
  if (ps.acc("DECLARE")) {  // MPSQL declaration section: name type [= expr]; ..
    while (!ps.end() && !ps.atKw("BEGIN")) {
      if (sym(r->toks[ps.p], ";")) { ++ps.p; continue; }
      Node d;
      d.k = Node::DECLARE;
      d.var = r->toks[ps.p++].v;
      while (!ps.end() && !sym(r->toks[ps.p], ";") && !sym(r->toks[ps.p], "=") && !sym(r->toks[ps.p], ":=") && !ps.atKw("DEFAULT")) ++ps.p;
      if (!ps.end() && !sym(r->toks[ps.p], ";")) {
        ++ps.p;
        d.b = ps.p;
        d.e = ps.until({});
      }
      ps.semi();
      r->decls.push_back(d);
    }
  }
  if (ps.atKw("BEGIN")) {
    r->body.push_back(ps.stmt());
  } else {
    r->body = ps.list();  // a single statement body (MySQL: RETURN expr)
  }
  // Inlineable function: the body is just RETURN expr.
  const std::vector<Node>* b = &r->body;
  if (b->size() == 1 && (*b)[0].k == Node::BLOCK && (*b)[0].bodies.size() == 1) b = &(*b)[0].bodies[0];
  if (r->isFunc && r->decls.empty() && b->size() == 1 && (*b)[0].k == Node::RETURN) {
    const Node& n = (*b)[0];
    if (n.e > n.b) r->inlineExpr = sql.substr(r->toks[n.b].b, r->toks[n.e - 1].e - r->toks[n.b].b);
  }
  return r;
}

class Interp {
 public:
  Interp(Engine& eng, Routine& r) : eng(eng), r(r) {}
  Engine& eng;
  Routine& r;
  std::map<std::string, Var> vars;
  std::map<std::string, std::map<std::string, Var>> rows;  // FOR r IN query: r -> column -> value
  ResultSet last;
  bool haveResult = false;
  std::vector<ResultSet> all;  // every result set the body produced, in order (MySQL CALL semantics)
  void produced(ResultSet rs) {
    std::vector<ResultSet> more = std::move(rs.extra);
    rs.extra.clear();
    last = rs;
    haveResult = true;
    all.push_back(std::move(rs));
    for (auto& m : more) last = m, all.push_back(std::move(m));
  }

  // SQL text of tokens [b, e) with variables replaced by literals.
  std::string text(size_t b, size_t e) {
    std::string o;
    const auto& t = r.toks;
    for (size_t i = b; i < e; ++i) {
      if (i > b) o += r.src.substr(t[i - 1].e, t[i].b - t[i - 1].e);
      const PTok& k = t[i];
      if (k.k == PTok::WORD) {
        const std::string lw = lower(k.v);
        const bool afterDot = i > b && sym(t[i - 1], ".");
        const bool call = i + 1 < e && sym(t[i + 1], "(");
        if (lw == "scalar" && call) continue;  // SCALAR(SELECT ..) is a scalar subquery
        if (!afterDot && i + 2 < e && sym(t[i + 1], ".") && rows.count(lw)) {
          auto& row = rows[lw];
          auto it = row.find(lower(t[i + 2].v));
          if (it != row.end()) {
            o += it->second.lit();
            i += 2;
            continue;
          }
        }
        if (!afterDot && !call && !(i + 1 < e && sym(t[i + 1], "."))) {
          auto it = vars.find(lw);
          if (it != vars.end()) {
            o += it->second.lit();
            continue;
          }
        }
      }
      o += r.src.substr(k.b, k.e - k.b);
    }
    return o;
  }
  static Var cell(const ResultSet& rs, size_t row, size_t col) {
    Var v;
    if (row >= rs.rows.size() || col >= rs.rows[row].size() || rs.rows[row][col] == kNullCell) return v;
    v.null = false;
    v.s = rs.rows[row][col];
    LType ty = col < rs.types.size() ? rs.types[col] : LType::VARCHAR;
    v.num = ty == LType::INT || ty == LType::BIGINT || ty == LType::DOUBLE || ty == LType::DECIMAL;
    return v;
  }
  Var eval(size_t b, size_t e) {
    if (e <= b) return Var{};
    ResultSet rs = eng.execute("SELECT " + text(b, e));
    return cell(rs, 0, 0);
  }
  bool cond(std::pair<size_t, size_t> c) {
    Var v = eval(c.first, c.second);
    if (v.null) return false;
    if (v.num) return std::strtod(v.s.c_str(), nullptr) != 0;
    return !v.s.empty() && v.s != "0";
  }
  void assign(const std::string& name, Var v) {
    std::string n = lower(name);
    if (!n.empty() && n[0] == '@') {  // user variable: SET @x = .. at the session
      setUserVar(name, v.null ? Value::Null() : v.num ? (v.s.find_first_of(".eE") != std::string::npos ? Value::D(std::strtod(v.s.c_str(), nullptr)) : Value::I(std::atoll(v.s.c_str()))) : Value::S(v.s));
      return;
    }
    vars[n] = v;
  }
  void run(const std::vector<Node>& body) {
    for (const Node& n : body) exec(n);
  }
  void loopBody(const std::vector<Node>& body, bool& exit) {
    try {
      run(body);
    } catch (ExitLoop&) {
      exit = true;
    } catch (ContinueLoop&) {
    }
  }
  void exec(const Node& n) {
    switch (n.k) {
      case Node::DECLARE: vars[lower(n.var)] = eval(n.b, n.e); break;
      case Node::ASSIGN: assign(n.var, eval(n.b, n.e)); break;
      case Node::IF: {
        for (size_t i = 0; i < n.conds.size(); ++i)
          if (cond(n.conds[i])) {
            run(n.bodies[i]);
            return;
          }
        if (n.hasElse) run(n.bodies.back());
        break;
      }
      case Node::WHILE: {
        bool exit = false;
        for (int64_t it = 0; !exit && cond(n.conds[0]); ++it) {
          if (it > 100'000'000) throw SqlError("loop limit exceeded", 1105);
          loopBody(n.bodies[0], exit);
        }
        break;
      }
      case Node::LOOP: {
        bool exit = false;
        for (int64_t it = 0; !exit; ++it) {
          if (it > 100'000'000) throw SqlError("loop limit exceeded", 1105);
          loopBody(n.bodies[0], exit);
        }
        break;
      }
      case Node::FOR_RANGE: {
        Var lo = eval(n.conds[0].first, n.conds[0].second), hi = eval(n.conds[1].first, n.conds[1].second);
        int64_t a = std::atoll(lo.s.c_str()), z = std::atoll(hi.s.c_str());
        bool exit = false;
        for (int64_t i = a; i <= z && !exit; ++i) {
          Var v;
          v.null = false;
          v.num = true;
          v.s = std::to_string(i);
          vars[lower(n.var)] = v;
          loopBody(n.bodies[0], exit);
        }
        break;
      }
      case Node::FOR_QUERY: {
        ResultSet rs = eng.execute(text(n.b, n.e));
        bool exit = false;
        for (size_t i = 0; i < rs.rows.size() && !exit; ++i) {
          auto& row = rows[lower(n.var)];
          row.clear();
          for (size_t c = 0; c < rs.names.size(); ++c) row[lower(rs.names[c])] = cell(rs, i, c);
          loopBody(n.bodies[0], exit);
        }
        rows.erase(lower(n.var));
        break;
      }
      case Node::EXIT:
        if (n.conds.empty() || cond(n.conds[0])) throw ExitLoop{};
        break;
      case Node::CONTINUE:
        if (n.conds.empty() || cond(n.conds[0])) throw ContinueLoop{};
        break;
      case Node::RETURN: throw ReturnNow{eval(n.b, n.e)};
      case Node::ECHO: {
        std::string q = text(n.b, n.e);
        // ECHO p(args): the result of another procedure
        ResultSet rs = eng.execute(kw(r.toks[n.b], "SELECT") || kw(r.toks[n.b], "WITH") || kw(r.toks[n.b], "(") ? q : "CALL " + q);
        if (rs.isResult) produced(std::move(rs));
        break;
      }
      case Node::RAISE: {
        Var v = eval(n.b + (n.e > n.b && kw(r.toks[n.b], "USER_EXCEPTION") ? 1 : 0), n.e);
        throw SqlError(v.null ? "user exception" : v.s, 1644);
      }
      case Node::BLOCK: {
        if (n.bodies.size() < 2) {
          run(n.bodies[0]);
          break;
        }
        try {
          run(n.bodies[0]);
        } catch (SqlError& e) {
          vars["__error"] = Var{false, false, e.what()};
          run(n.bodies[1]);
        }
        break;
      }
      case Node::SQL: {
        if (!n.intoVars.empty()) {  // SELECT .. INTO vars: drop the INTO clause, assign the first row
          size_t a = std::stoul(n.var.substr(0, n.var.find(':'))), z = std::stoul(n.var.substr(n.var.find(':') + 1));
          std::string q = text(n.b, a) + " " + text(z, n.e);
          ResultSet rs = eng.execute(q);
          for (size_t i = 0; i < n.intoVars.size(); ++i) assign(n.intoVars[i], cell(rs, 0, i));
          break;
        }
        ResultSet rs = eng.execute(text(n.b, n.e));
        if (rs.isResult) produced(std::move(rs));
        break;
      }
      default: break;
    }
  }
};

}  // namespace

// FROM fn(args) of a table-valued function (CREATE FUNCTION .. RETURNS TABLE AS RETURN SELECT ..):
// its SELECT with the parameters replaced by the arguments.
std::shared_ptr<Stmt> tvfStatement(const std::string& name, const std::vector<ExprP>& args) {
  std::shared_ptr<Routine> r;
  {
    std::lock_guard<std::mutex> g(gRoutinesMu);
    auto it = gRoutines.find(lower(name.substr(name.find('.') == std::string::npos ? 0 : name.find('.') + 1)));
    if (it == gRoutines.end() || !it->second->isFunc) throw SqlError("FUNCTION " + name + " does not exist", 1305);
    r = it->second;
  }
  if (r->inlineExpr.empty() || upper(r->inlineExpr.substr(0, 6)) != "SELECT")
    throw SqlError("FUNCTION " + r->name + " is not a table-valued function (RETURNS TABLE AS RETURN SELECT ..)", 1235);
  if (args.size() != r->params.size())
    throw SqlError("Incorrect number of arguments for FUNCTION " + r->name + "; expected " + std::to_string(r->params.size()), 1318);
  auto s = std::make_shared<Stmt>(parseSql(r->inlineExpr));
  std::function<void(ExprP&)> sub = [&](ExprP& x) {
    if (!x) return;
    if (x->kind == Expr::COL && x->table.empty())
      for (size_t i = 0; i < r->params.size(); ++i)
        if (lower(x->name) == r->params[i]) { x = args[i]; return; }
    for (auto& a : x->args) sub(a);
  };
  for (Stmt* st = s.get(); st; st = st->unionNext.get()) {
    for (auto& it : st->items) sub(it.e);
    sub(st->where);
    sub(st->having);
    for (auto& g : st->groupBy) sub(g);
    for (auto& o : st->orderBy) sub(o.e);
    for (auto& j : st->joinConds) sub(j);
    for (auto& f : st->from) sub(f.on);
  }
  return s;
}

// A value as SQL literal text (procedure / function arguments).
std::string udfLiteral(const Value& v) {
  if (v.isNull()) return "NULL";
  if (v.kind == Value::STR || v.kind == Value::DATE || v.kind == Value::DATETIME) {
    std::string s = v.toString(), o = "'";
    for (char ch : s) o += ch == '\'' ? std::string("''") : ch == '\\' ? std::string("\\\\") : std::string(1, ch);
    return o + "'";
  }
  return v.toString();
}

namespace {

// A function call in a query: inline single-RETURN functions, evaluate others with constant arguments.
ExprP inlineUdf(const ExprP& e) {
  std::shared_ptr<Routine> r;
  {
    std::lock_guard<std::mutex> g(gRoutinesMu);
    if (gRoutines.empty()) return nullptr;
    auto it = gRoutines.find(lower(e->op));
    if (it == gRoutines.end() || !it->second->isFunc) return nullptr;
    r = it->second;
  }
  if (e->args.size() != r->params.size())
    throw SqlError("Incorrect number of arguments for FUNCTION " + r->name + "; expected " + std::to_string(r->params.size()), 1318);
  if (tCallDepth > 32) throw SqlError("function recursion too deep: " + r->name, 1456);
  if (!r->inlineExpr.empty()) {
    ++tCallDepth;
    Stmt s;
    try {
      s = parseSql("SELECT " + r->inlineExpr);
    } catch (...) {
      --tCallDepth;
      throw;
    }
    --tCallDepth;
    std::function<ExprP(const ExprP&)> sub = [&](const ExprP& x) -> ExprP {
      if (!x) return x;
      if (x->kind == Expr::COL && x->table.empty())
        for (size_t i = 0; i < r->params.size(); ++i)
          if (lower(x->name) == r->params[i]) return e->args[i];
      for (auto& a : x->args) a = sub(a);
      return x;
    };
    return sub(s.items[0].e);
  }
  std::vector<std::string> lits;
  bool constant = true;
  for (auto& a : e->args) constant = constant && a->kind == Expr::LIT;
  if (!constant) {  // over columns: a host function running the body once per distinct argument tuple
    auto n = std::make_shared<Expr>();
    n->kind = Expr::FUNC;
    n->op = "#UDF";
    static std::atomic<uint64_t> gen{0};  // per-call-site memo of results (see evalHostFunc)
    n->args = {mkLit(Value::S(r->name)), mkLit(Value::S(r->retType + "#" + std::to_string(++gen)))};
    for (auto& a : e->args) n->args.push_back(a);
    return n;
  }
  for (auto& a : e->args) lits.push_back(udfLiteral(a->lit));
  Value out = gEngine->callFunction(r->name, lits);
  return mkLit(out);
}

}  // namespace

Value callUdf(const std::string& name, const std::vector<std::string>& args) { return gEngine->callFunction(name, args); }

Value userVar(const std::string& name) {
  auto it = tUserVars.find(lower(name));
  return it == tUserVars.end() ? Value::Null() : it->second;
}
void setUserVar(const std::string& name, Value v) { tUserVars[lower(name)] = std::move(v); }

Value Engine::callFunction(const std::string& name, const std::vector<std::string>& args) {
  std::string q = "CALL " + name + "(";
  for (size_t i = 0; i < args.size(); ++i) q += (i ? "," : "") + args[i];
  ResultSet rs = execute(q + ")");
  if (rs.rows.empty() || rs.rows[0].empty() || rs.rows[0][0] == kNullCell) return Value::Null();
  const std::string& s = rs.rows[0][0];
  LType t = rs.types.empty() ? LType::VARCHAR : rs.types[0];
  if (t == LType::INT || t == LType::BIGINT) return Value::I(std::atoll(s.c_str()));
  if (t == LType::DOUBLE || t == LType::DECIMAL) return Value::D(std::strtod(s.c_str(), nullptr));
  return Value::S(s);
}

std::vector<std::string> Engine::routineDefs() {
  std::lock_guard<std::mutex> g(gRoutinesMu);
  std::vector<std::string> out;
  for (auto& [k, r] : gRoutines) out.push_back(r->createSql);
  return out;
}

// CREATE / DROP / CALL / ECHO / SHOW for procedures and functions; false when sql is something else.
bool Engine::routineStatement(const std::string& sql, ResultSet& out) {
  static const std::regex createRe(R"(^\s*CREATE\s+(OR\s+REPLACE\s+)?(DEFINER\s*=\s*\S+\s+)?(PROCEDURE|FUNCTION)\b)", std::regex::icase);
  static const std::regex dropRe(R"(^\s*DROP\s+(PROCEDURE|FUNCTION)\s+(IF\s+EXISTS\s+)?`?([A-Za-z0-9_.]+)`?\s*;?\s*$)", std::regex::icase);
  static const std::regex callRe(R"(^\s*(CALL|ECHO)\s+`?([A-Za-z0-9_.]+)`?\s*(\(([\s\S]*)\))?\s*;?\s*$)", std::regex::icase);
  static const std::regex showRe(R"(^\s*SHOW\s+(PROCEDURE|FUNCTION)\s+STATUS\b|^\s*SHOW\s+(PROCEDURES|FUNCTIONS)\b)", std::regex::icase);
  if (!gEngine) {
    gEngine = this;
    udfHook = inlineUdf;
  }
  static const std::regex setVarRe(R"(^\s*SET\s+(@[A-Za-z0-9_$]+)\s*:?=\s*([\s\S]*?)\s*;?\s*$)", std::regex::icase);
  std::smatch m;
  // SQL-level prepared statements: PREPARE s FROM 'sql' | @var; EXECUTE s [USING @a, ..]; DEALLOCATE PREPARE s.
  static const std::regex prepRe(R"(^\s*PREPARE\s+`?([A-Za-z0-9_]+)`?\s+FROM\s+([\s\S]+?)\s*;?\s*$)", std::regex::icase);
  static const std::regex execRe(R"(^\s*EXECUTE\s+`?([A-Za-z0-9_]+)`?(\s+USING\s+([\s\S]+?))?\s*;?\s*$)", std::regex::icase);
  static const std::regex deallocRe(R"(^\s*(DEALLOCATE|DROP)\s+PREPARE\s+`?([A-Za-z0-9_]+)`?\s*;?\s*$)", std::regex::icase);
  thread_local std::map<std::string, std::string> prepared;
  {
    static const std::set<std::string> kw = {"CREATE", "DROP", "CALL", "ECHO", "SHOW", "SET", "PREPARE", "EXECUTE", "DEALLOCATE"};
    if (!kw.count(firstWord(sql))) return false;  // std::regex_search would scan the whole (maybe huge) statement
  }
  if (std::regex_match(sql, m, prepRe)) {
    std::string src = m[2].str();
    if (!src.empty() && src[0] == '@') {
      src = userVar(src).toString();
    } else if (src.size() >= 2 && (src[0] == '\'' || src[0] == '"')) {
      std::string v;
      for (size_t i = 1; i + 1 < src.size(); ++i) {
        if (src[i] == '\\' && i + 2 < src.size()) { v += src[++i]; continue; }
        if (src[i] == src[0] && src[i + 1] == src[0]) { v += src[i++]; continue; }
        v += src[i];
      }
      src = v;
    } else {
      throw SqlError("PREPARE needs a string literal or a @variable", 1064);
    }
    {  // syntax check now, with the placeholders as NULL
      std::string chk;
      char quote = 0;
      for (char ch : src) {
        if (quote) { chk += ch; if (ch == quote) quote = 0; continue; }
        if (ch == '\'' || ch == '"' || ch == '`') quote = ch;
        chk += ch == '?' && !quote ? std::string("NULL") : std::string(1, ch);
      }
      parseSql(chk);
    }
    prepared[lower(m[1].str())] = src;
    out = ResultSet{};
    out.info = "Statement prepared";
    return true;
  }
  if (std::regex_match(sql, m, execRe)) {
    auto it = prepared.find(lower(m[1].str()));
    if (it == prepared.end()) throw SqlError("Unknown prepared statement handler (" + m[1].str() + ") given to EXECUTE", 1243);
    std::vector<Value> args;
    if (m[3].matched) {
      std::string list = m[3].str();
      size_t a = 0;
      while (a <= list.size()) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) b = list.size();
        std::string v = list.substr(a, b - a);
        v.erase(0, v.find_first_not_of(" \t"));
        v.erase(v.find_last_not_of(" \t") + 1);
        if (!v.empty()) args.push_back(userVar(v));
        a = b + 1;
      }
    }
    // Substitute ? placeholders (outside quotes) with the values as SQL literals.
    const std::string& text = it->second;
    std::string q;
    size_t argi = 0;
    char quote = 0;
    for (size_t i = 0; i < text.size(); ++i) {
      const char c = text[i];
      if (quote) {
        q += c;
        if (c == '\\' && i + 1 < text.size()) q += text[++i];
        else if (c == quote) quote = 0;
        continue;
      }
      if (c == '\'' || c == '"' || c == '`') { quote = c; q += c; continue; }
      if (c != '?') { q += c; continue; }
      if (argi >= args.size()) throw SqlError("Incorrect arguments to EXECUTE", 1210);
      const Value& v = args[argi++];
      if (v.isNull()) q += "NULL";
      else if (v.kind == Value::STR) {
        q += '\'';
        for (char ch : v.s) q += ch == '\'' ? std::string("''") : ch == '\\' ? std::string("\\\\") : std::string(1, ch);
        q += '\'';
      } else {
        q += v.toString();
      }
    }
    if (argi != args.size()) throw SqlError("Incorrect arguments to EXECUTE", 1210);
    out = execute(q);
    return true;
  }
  if (std::regex_match(sql, m, deallocRe)) {
    prepared.erase(lower(m[2].str()));
    out = ResultSet{};
    return true;
  }
  if (std::regex_match(sql, m, setVarRe)) {
    ResultSet rs = execute("SELECT " + m[2].str());
    Var v = Interp::cell(rs, 0, 0);
    setUserVar(m[1].str(), v.null ? Value::Null() : v.num ? (v.s.find_first_of(".eE") != std::string::npos ? Value::D(std::strtod(v.s.c_str(), nullptr)) : Value::I(std::atoll(v.s.c_str()))) : Value::S(v.s));
    out = ResultSet{};
    return true;
  }
  if ((std::regex_search(sql, m, createRe) || std::regex_search(sql, m, dropRe)) && isReplica() && !tApplying)
    throw SqlError("The server is a read-only replica (--replica-of); run writes on the primary or PROMOTE it", 1290);
  if (std::regex_search(sql, m, createRe)) {
    if (sessionUser() != "root") {  // routine management needs SUPER, like other account-level DDL
      Stmt probe;
      probe.type = Stmt::PIPELINE;
      checkPrivileges(probe);
    }
    auto r = parseRoutine(sql);
    std::string key = lower(r->name);
    {
      std::lock_guard<std::mutex> g(gRoutinesMu);
      if (gRoutines.count(key) && !m[1].matched)
        throw SqlError(std::string(r->isFunc ? "FUNCTION " : "PROCEDURE ") + r->name + " already exists", 1304);
      gRoutines[key] = r;
    }
    logWrite(sql);
    out = ResultSet{};
    return true;
  }
  if (std::regex_search(sql, m, dropRe)) {
    std::string name = m[3].str();
    if (name.find('.') != std::string::npos) name = name.substr(name.find('.') + 1);
    {
      std::lock_guard<std::mutex> g(gRoutinesMu);
      if (!gRoutines.erase(lower(name)) && !m[2].matched)
        throw SqlError(upper(m[1].str()) + " " + name + " does not exist", 1305);
    }
    logWrite(sql);
    out = ResultSet{};
    return true;
  }
  if (std::regex_search(sql, m, showRe)) {
    out = ResultSet{};
    out.isResult = true;
    out.names = {"Db", "Name", "Type", "Definer"};
    out.types = {LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::VARCHAR};
    std::lock_guard<std::mutex> g(gRoutinesMu);
    const std::string kind = upper(m[1].matched ? m[1].str() : m[2].str());
    bool funcs = kind[0] == 'F';
    for (auto& [k, r] : gRoutines)
      if (r->isFunc == funcs) out.rows.push_back({currentDb(), r->name, r->isFunc ? "FUNCTION" : "PROCEDURE", "root@%"});
    return true;
  }
  if (!std::regex_match(sql, m, callRe)) return false;
  std::string name = m[2].str();
  if (name.find('.') != std::string::npos) name = name.substr(name.find('.') + 1);
  std::shared_ptr<Routine> r;
  {
    std::lock_guard<std::mutex> g(gRoutinesMu);
    auto it = gRoutines.find(lower(name));
    if (it == gRoutines.end()) {
      if (upper(m[1].str()) == "ECHO") return false;
      throw SqlError("PROCEDURE " + name + " does not exist", 1305);
    }
    r = it->second;
  }
  // Arguments: evaluate each top-level comma-separated expression.
  std::vector<Var> args;
  if (m[4].matched && m[4].str().find_first_not_of(" \t\r\n") != std::string::npos) {
    std::string a = m[4].str();
    auto toks = lex(a);
    int depth = 0;
    size_t start = 0;
    auto flush = [&](size_t end) {
      std::string ex = a.substr(toks[start].b, toks[end - 1].e - toks[start].b);
      ResultSet rs = execute("SELECT " + ex);
      args.push_back(Interp::cell(rs, 0, 0));
    };
    for (size_t i = 0; i < toks.size(); ++i) {
      if (sym(toks[i], "(")) ++depth;
      else if (sym(toks[i], ")")) --depth;
      else if (depth == 0 && sym(toks[i], ",")) {
        flush(i);
        start = i + 1;
      }
    }
    flush(toks.size());
  }
  if (args.size() != r->params.size())
    throw SqlError("Incorrect number of arguments for PROCEDURE " + r->name + "; expected " + std::to_string(r->params.size()) +
                       ", got " + std::to_string(args.size()), 1318);
  if (++tCallDepth > 32) {
    --tCallDepth;
    throw SqlError("procedure recursion too deep: " + r->name, 1456);
  }
  struct Depth { ~Depth() { --tCallDepth; } } depthGuard;
  Interp in(*this, *r);
  for (size_t i = 0; i < args.size(); ++i) in.vars[r->params[i]] = args[i];
  for (auto& d : r->decls) in.exec(d);
  try {
    in.run(r->body);
  } catch (ReturnNow& ret) {
    out = ResultSet{};
    out.isResult = true;
    out.names = {r->name};
    out.types = {ret.v.null || !ret.v.num ? LType::VARCHAR
                 : ret.v.s.find_first_of(".eE") != std::string::npos ? LType::DOUBLE : LType::BIGINT};
    out.rows = {{ret.v.null ? kNullCell : ret.v.s}};
    return true;
  } catch (ExitLoop&) {
  }
  out = ResultSet{};
  if (!in.all.empty()) {
    out = std::move(in.all[0]);
    for (size_t i = 1; i < in.all.size(); ++i) out.extra.push_back(std::move(in.all[i]));
  }
  return true;
}

}  // namespace gdb
