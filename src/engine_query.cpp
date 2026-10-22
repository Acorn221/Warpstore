#include "engine_core.h"

namespace gdb {
// ================================================================== query layer
// CTEs, derived tables, subqueries, UNION and COUNT(DISTINCT) on top of doSelect: inner queries run
// first and their results become query-scoped GPU temp tables (or literals / value sets).


ExprP cloneExpr(const ExprP& e) {
  if (!e) return e;
  auto y = std::make_shared<Expr>(*e);
  y->bound = false;
  y->bt = y->bc = -1;
  for (auto& a : y->args) a = cloneExpr(a);
  for (auto& a : y->partBy) a = cloneExpr(a);
  for (auto& o : y->winOrder) o.first = cloneExpr(o.first);
  if (e->sub) y->sub = std::make_shared<Stmt>(*e->sub);  // statements are cloned again before use
  return y;
}

Stmt cloneStmt(const Stmt& s) {
  Stmt c = s;
  for (auto& it : c.items) it.e = cloneExpr(it.e);
  for (auto& f : c.from) {
    f.on = cloneExpr(f.on);
    if (f.sub) f.sub = std::make_shared<Stmt>(cloneStmt(*f.sub));
  }
  for (auto& j : c.joinConds) j = cloneExpr(j);
  c.where = cloneExpr(c.where);
  c.having = cloneExpr(c.having);
  for (auto& g : c.groupBy) g = cloneExpr(g);
  for (auto& o : c.orderBy) o.e = cloneExpr(o.e);
  if (c.unionNext) c.unionNext = std::make_shared<Stmt>(cloneStmt(*c.unionNext));
  for (auto& [n, q] : c.ctes) q = std::make_shared<Stmt>(cloneStmt(*q));
  return c;
}

void walkStmtExprs(Stmt& s, const std::function<void(ExprP&)>& fn) {
  for (auto& it : s.items) fn(it.e);
  if (s.where) fn(s.where);
  if (s.having) fn(s.having);
  for (auto& j : s.joinConds) fn(j);
  for (auto& f : s.from)
    if (f.on) fn(f.on);
  for (auto& g : s.groupBy) fn(g);
  for (auto& o : s.orderBy) fn(o.e);
}

void formatRows(ResultSet& r) {
  if (!r.typed) return;
  for (auto& v : r.vals) {
    std::vector<std::string> row;
    for (auto& x : v) row.push_back(cell(x));
    r.rows.push_back(std::move(row));
  }
  r.vals.clear();
  r.typed = false;
}

std::string Engine::makeTemp(const ResultSet& r, QueryCtx& q, const std::string& hint) {
  static std::atomic<int> seq{0};
  std::string name = "__tmp" + std::to_string(++seq) + "_" + hint;
  std::vector<ColumnMeta> cols;
  std::set<std::string> seen;
  for (size_t i = 0; i < r.names.size(); ++i) {
    std::string n = r.names[i];
    for (int k = 2; seen.count(lower(n)); ++k) n = r.names[i] + "_" + std::to_string(k);
    seen.insert(lower(n));
    ColumnMeta m;
    m.name = n;
    m.lt = i < r.types.size() ? r.types[i] : LType::VARCHAR;
    if (m.lt == LType::DECIMAL) {  // the scale travels with the values (Value::scale); none known: DOUBLE
      int sc = -1;
      for (auto& row : r.vals)
        if (i < row.size() && !row[i].isNull() && row[i].scale >= 0) sc = std::max(sc, (int)row[i].scale);
      if (sc < 0) m.lt = LType::DOUBLE;
      else m.scale = sc;
    }
    if (m.lt == LType::VECTOR) {  // the dimension comes from the values (all NULL: kept as text)
      m.lt = LType::VARCHAR;
      for (auto& row : r.vals) {
        std::vector<float> f;
        if (i < row.size() && !row[i].isNull() && parseVector(row[i].toString(), f) && !f.empty()) {
          m.lt = LType::VECTOR;
          m.vecDim = (int)f.size();
          break;
        }
      }
    }
    m.pt = physOf(m.lt);
    cols.push_back(m);
  }
  if (cols.empty()) throw SqlError("derived table has no columns", 1105);
  if (r.insertCols)  // typed columns (a projection): DECIMAL scales from the result
    for (size_t i = 0; i < cols.size() && i < r.colScales.size(); ++i)
      if (r.types[i] == LType::DECIMAL) {
        cols[i].lt = r.colScales[i] >= 0 ? LType::DECIMAL : LType::DOUBLE;
        cols[i].scale = std::max(0, r.colScales[i]);
        cols[i].pt = physOf(cols[i].lt);
      }
  auto t = std::make_shared<Table>(name, cols, -1, -1, std::vector<int>{}, std::vector<int>{}, 1, false);
  t->db = "__tmp";  // (before the rows: temporary tables stay out of the session's transaction)
  if (r.insertCols) {  // a projection's columns: appended as they are (strings keep their codes and dictionary)
    const size_t nc = cols.size();
    int64_t n = 0;
    for (size_t i = 0; i < nc; ++i) n = std::max<int64_t>(n, !r.intCols[i].empty() ? (int64_t)r.intCols[i].size() : (int64_t)r.numCols[i].size());
    std::vector<std::vector<int64_t>> iv(nc);
    std::vector<std::vector<double>> dv(nc);
    for (size_t i = 0; i < nc; ++i) {
      const ColumnMeta& m = cols[i];
      if (m.lt == LType::VARCHAR) t->dicts[i] = r.colDicts[i];
      if (!r.intCols[i].empty()) {
        iv[i] = r.intCols[i];
        continue;
      }
      const std::vector<double>& x = r.numCols[i];
      if (m.pt == PType::F64) {
        dv[i] = x;
        continue;
      }
      const int64_t nul = m.pt == PType::I64 ? kNullI64 : kNullI32;
      const double mul = m.lt == LType::DECIMAL ? pow10i(m.scale) : 1.0;
      iv[i].resize(x.size());
      for (size_t k = 0; k < x.size(); ++k) iv[i][k] = x[k] != x[k] ? nul : std::llround(x[k] * mul);
    }
    for (size_t i = 0; i < nc; ++i) t->hasNull[i] = 1;
    if (n > 0) {
      t->appendColumns(iv, dv, n);
      t->flush(true);
    }
    std::lock_guard<std::mutex> g(cat_.mu);
    cat_.tables["__tmp." + lower(name)] = t;
    q.temps.push_back("__tmp." + lower(name));
    return "__tmp." + name;
  }
  if (!r.devCols.empty()) {  // built on the GPU: the columns become the table's one segment (no host copy)
    const int64_t n = r.devRows;
    const size_t nc = cols.size();
    auto seg = std::make_unique<Segment>();
    seg->part = 0;
    seg->n = n;
    seg->host.resize(nc);  // (filled on demand: Table::colData)
    seg->dev.resize(nc, nullptr);
    seg->mapped.assign(nc, 0);
    seg->enc.assign(nc, 0);
    seg->encHost.resize(nc);
    seg->mn.resize(nc);
    seg->mx.resize(nc);
    for (size_t c = 0; c < nc; ++c) {
      seg->dev[c] = r.devCols[c]->p;
      r.devCols[c]->p = nullptr;
      seg->mn[c] = r.devCols[c]->mn;
      seg->mx[c] = r.devCols[c]->mx;
      seg->devBytes += n * t->colBytes((int)c);
      if (cols[c].lt == LType::VARCHAR) t->dicts[c] = r.colDicts[c];  // (shared: temp tables add no strings)
      t->hasNull[c] = 1;
    }
    seg->del->host.assign((size_t)(n + 31) / 32, 0);
    if (n > 0) t->segs.push_back(std::move(seg));
    std::lock_guard<std::mutex> g(cat_.mu);
    cat_.tables["__tmp." + lower(name)] = t;
    q.temps.push_back("__tmp." + lower(name));
    return "__tmp." + name;
  }
  if (!r.numCols.empty()) {  // numeric columns (a GROUP BY result): straight into the write buffer
    const size_t n = r.numCols[0].size();
    WriteBuffer& b = *t->bufs[0];
    for (size_t c = 0; c < cols.size(); ++c) {
      const ColumnMeta& m = cols[c];
      if (m.lt == LType::VARCHAR) {  // dictionary codes: the column takes a copy of their dictionary
        if (c >= r.colDicts.size() || !r.colDicts[c]) throw SqlError("internal: string column without a dictionary", 1105);
        t->dicts[c] = r.colDicts[c];  // (shared: temp tables add no strings)
      }
      const int sz = t->colBytes((int)c);
      std::vector<uint8_t> bytes(n * sz);
      std::atomic<bool> nul{false};
      parallelFor(n, [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
          const double d = r.numCols[c][i];
          if (m.pt == PType::F64) { memcpy(&bytes[i * 8], &d, 8); if (d != d) nul = true; continue; }
          int64_t x = d != d ? (m.pt == PType::I64 ? kNullI64 : kNullI32) : std::llround(d);
          if (d != d) nul = true;
          if (m.pt == PType::I64) memcpy(&bytes[i * 8], &x, 8);
          else { const int32_t y = (int32_t)x; memcpy(&bytes[i * 4], &y, 4); }
        }
      });
      if (nul) t->hasNull[c] = 1;
      b.cols[c] = std::move(bytes);
    }
    b.n = (int64_t)n;
    b.del = std::make_shared<DelBits>();
    b.del->host.assign((n + 31) / 32, 0);
    ++b.version;
    ++b.gen;
    t->flush(true);
    std::lock_guard<std::mutex> g(cat_.mu);
    cat_.tables["__tmp." + lower(name)] = t;
    q.temps.push_back("__tmp." + lower(name));
    return "__tmp." + name;
  }
  bool bulk = true;
  for (auto& c : cols) bulk = bulk && c.pt != PType::VEC;
  if (bulk) {  // column by column straight into the write buffer (on all cores), then one flush
    const size_t n = r.vals.size(), nc = cols.size();
    WriteBuffer& b = *t->bufs[0];
    for (size_t c = 0; c < nc; ++c) {
      const ColumnMeta& m = cols[c];
      const int sz = t->colBytes((int)c);
      std::vector<uint8_t> bytes(n * sz);
      std::atomic<bool> nul{false};
      if (m.lt == LType::VARCHAR) {
        Dict& d = *t->dicts[c];
        for (size_t i = 0; i < n; ++i) {
          const Value& v = r.vals[i][c];
          int32_t code = kNullI32;
          if (v.isNull()) nul = true;
          else code = d.getOrAdd(v.kind == Value::STR ? v.s : v.toString());
          memcpy(&bytes[i * 4], &code, 4);
        }
      } else {
        const double unscale = m.lt == LType::DECIMAL ? pow10i(m.scale) : 1.0;
        parallelFor(n, [&](size_t lo, size_t hi) {
          for (size_t i = lo; i < hi; ++i) {
            const Value& v = r.vals[i][c];
            if (m.pt == PType::F64) {
              const double d = v.isNull() ? nullv() : v.kind == Value::STR ? std::strtod(v.s.c_str(), nullptr) : v.num();
              if (v.isNull()) nul = true;
              memcpy(&bytes[i * 8], &d, 8);
              continue;
            }
            int64_t x;
            if (v.isNull()) {
              nul = true;
              x = m.pt == PType::I64 ? kNullI64 : kNullI32;
            } else if (m.lt == LType::DECIMAL) {
              x = std::llround(v.num() * unscale);
            } else if (m.lt == LType::DATE && v.kind == Value::DATETIME) {
              x = floorDiv(v.i, kUsPerDay);
            } else if (m.lt == LType::DATETIME && v.kind == Value::DATE) {
              x = v.i * kUsPerDay;
            } else if (v.kind == Value::STR) {
              x = std::strtoll(v.s.c_str(), nullptr, 10);
            } else {
              x = v.kind == Value::DBL ? std::llround(v.d) : v.i;
            }
            if (m.pt == PType::I64) memcpy(&bytes[i * 8], &x, 8);
            else { const int32_t y = (int32_t)x; memcpy(&bytes[i * 4], &y, 4); }
          }
        });
      }
      if (nul) t->hasNull[c] = 1;
      b.cols[c] = std::move(bytes);
    }
    b.n = (int64_t)n;
    b.del = std::make_shared<DelBits>();
    b.del->host.assign((n + 31) / 32, 0);
    ++b.version;
    ++b.gen;
  } else {
    for (auto& row : r.vals) t->appendRow(row);
  }
  t->flush(true);
  std::lock_guard<std::mutex> g(cat_.mu);
  cat_.tables["__tmp." + lower(name)] = t;
  q.temps.push_back("__tmp." + lower(name));
  return "__tmp." + name;
}

void Engine::dropTemps(QueryCtx& q) {
  std::lock_guard<std::mutex> g(cat_.mu);
  for (auto& k : q.temps) cat_.tables.erase(k);
  q.temps.clear();
}

namespace {
// Column names of a FROM entry (catalog tables only; derived tables unknown here).
bool fromHasColumn(Catalog& cat, const Stmt& s, const std::string& col) {
  for (auto& f : s.from) {
    if (f.sub) continue;
    std::shared_ptr<Table> t;
    {
      std::lock_guard<std::mutex> g(cat.mu);
      auto it = cat.tables.find(Catalog::key(f.name));
      if (it != cat.tables.end()) t = it->second;
    }
    if (t && t->colIndex(col) >= 0) return true;
  }
  return false;
}
bool fromHasAlias(const Stmt& s, const std::string& a) {
  for (auto& f : s.from)
    if (strcasecmp(f.alias.c_str(), a.c_str()) == 0 || (f.alias.empty() && strcasecmp(f.name.c_str(), a.c_str()) == 0))
      return true;  // (an aliased table is only visible by its alias)
  return false;
}
}  // namespace

// Is alias a defined anywhere in s, including its derived tables and subqueries in FROM?
static bool fromHasAliasDeep(const Stmt& s, const std::string& a) {
  if (fromHasAlias(s, a)) return true;
  for (auto& f : s.from)
    if (f.sub && fromHasAliasDeep(*f.sub, a)) return true;
  return s.unionNext && fromHasAliasDeep(*s.unionNext, a);
}
// Expressions of s and of the derived tables in its FROM (correlated references may sit in there).
void walkStmtExprsDeep(Stmt& s, const std::function<void(ExprP&)>& fn) {
  walkStmtExprs(s, fn);
  for (auto& f : s.from)
    if (f.sub) walkStmtExprsDeep(*f.sub, fn);
  if (s.unionNext) walkStmtExprsDeep(*s.unionNext, fn);
}

// Does `col` (inside subquery `sub`) refer to the outer statement?
static bool outerRef(Catalog& cat, const ExprP& col, const Stmt& sub, const Stmt& outer,
                     const std::map<std::string, std::string>& ctes) {
  if (col->kind != Expr::COL) return false;
  if (!col->table.empty()) return !fromHasAliasDeep(sub, col->table) && fromHasAlias(outer, col->table);
  // Unqualified: outer if no table of the subquery has it but an outer table does.
  Stmt subR = sub;
  for (auto& f : subR.from) {
    auto it = ctes.find(lower(f.name));
    if (it != ctes.end()) f.name = it->second;
  }
  for (auto& f : subR.from)
    if (f.sub) return false;  // unknown columns: assume local
  return !fromHasColumn(cat, subR, col->name) && fromHasColumn(cat, outer, col->name);
}

void Engine::resolveSubqueries(Stmt& s, QueryCtx& q) {
  auto correlated = [&](Stmt& sub) {
    bool any = false;
    std::function<void(const ExprP&)> scan = [&](const ExprP& e) {
      if (!e || any) return;
      if (outerRef(cat_, e, sub, s, q.ctes)) any = true;
      for (auto& a : e->args) scan(a);
    };
    walkStmtExprsDeep(sub, [&](ExprP& e) { scan(e); });
    return any;
  };
  // EXISTS (SELECT .. WHERE inner = outer AND ..) -> outer IN (SELECT inner FROM .. WHERE ..)
  auto decorrelateExists = [&](ExprP& e) -> bool {
    Stmt sub = cloneStmt(*e->sub);
    std::vector<ExprP> conj;
    std::function<void(const ExprP&)> flat = [&](const ExprP& x) {
      if (x && x->kind == Expr::BIN && x->op == "AND") { flat(x->args[0]); flat(x->args[1]); }
      else if (x) conj.push_back(x);
    };
    flat(sub.where);
    ExprP outerCol, innerCol;
    std::vector<ExprP> rest;
    for (auto& c : conj) {
      if (!outerCol && c->kind == Expr::BIN && c->op == "=" && c->args[0]->kind == Expr::COL && c->args[1]->kind == Expr::COL) {
        bool o0 = outerRef(cat_, c->args[0], sub, s, q.ctes), o1 = outerRef(cat_, c->args[1], sub, s, q.ctes);
        if (o0 != o1) {
          outerCol = o0 ? c->args[0] : c->args[1];
          innerCol = o0 ? c->args[1] : c->args[0];
          continue;
        }
      }
      rest.push_back(c);
    }
    if (!outerCol) return false;
    // NULLs never satisfy the equality: inner NULLs are dropped and a NULL outer value gives false
    // (not NULL, which NOT EXISTS would keep as NULL).
    auto notNull = [](ExprP x) {
      auto n = std::make_shared<Expr>();
      n->kind = Expr::FUNC;
      n->op = "ISNOTNULL";
      n->args = {x};
      return n;
    };
    ExprP w = notNull(cloneExpr(innerCol));
    for (auto& r : rest) w = mkBin("AND", w, r);
    sub.where = w;
    sub.items = {SelectItem{innerCol, ""}};
    sub.orderBy.clear();
    sub.limit = -1;
    if (correlated(sub)) return false;
    auto in = std::make_shared<Expr>();
    in->kind = Expr::INLIST;
    in->args = {cloneExpr(outerCol)};
    in->sub = std::make_shared<Stmt>(std::move(sub));
    e = mkBin("AND", notNull(cloneExpr(outerCol)), in);
    return true;
  };
  // Any other correlated scalar subquery: evaluate it once per distinct combination of the outer
  // columns it references (from the outer FROM), keep the results in a temporary table and LEFT JOIN
  // it on those columns.
  auto applyPerOuterValue = [&](ExprP& e) {
    Stmt sub = cloneStmt(*e->sub);
    std::vector<ExprP> outerCols;
    std::set<std::string> seen;
    std::function<void(const ExprP&)> collect = [&](const ExprP& x) {
      if (!x) return;
      if (x->kind == Expr::COL && outerRef(cat_, x, sub, s, q.ctes)) {
        if (seen.insert(lower(x->table + "." + x->name)).second) outerCols.push_back(x);
        return;
      }
      for (auto& a : x->args) collect(a);
    };
    walkStmtExprsDeep(sub, [&](ExprP& x) { collect(x); });
    Stmt dq = cloneStmt(s);
    dq.items.clear();
    for (auto& c : outerCols) dq.items.push_back(SelectItem{cloneExpr(c), ""});
    dq.distinct = true;
    dq.where.reset();
    if (s.from.size() == 1) {  // one outer table: only the outer values its other filters let through
      std::function<bool(const ExprP&)> hasSub = [&](const ExprP& x) {
        if (!x) return false;
        if (x->sub) return true;
        for (auto& a : x->args)
          if (hasSub(a)) return true;
        return false;
      };
      std::function<void(const ExprP&)> keep = [&](const ExprP& x) {
        if (!x) return;
        if (x->kind == Expr::BIN && x->op == "AND") {
          keep(x->args[0]);
          keep(x->args[1]);
        } else if (!hasSub(x)) {
          dq.where = dq.where ? mkBin("AND", dq.where, cloneExpr(x)) : cloneExpr(x);
        }
      };
      keep(s.where);
    }
    dq.having.reset();
    dq.groupBy.clear();
    dq.orderBy.clear();
    dq.limit = -1;
    dq.offset = 0;
    dq.unionNext.reset();
    ResultSet keys = query(dq, q, false, true);
    if (keys.vals.size() > 20000)
      throw SqlError("correlated subquery: " + std::to_string(keys.vals.size()) + " distinct outer values (at most 20000)", 1235);
    ResultSet tab;
    tab.typed = true;
    for (size_t k = 0; k < outerCols.size(); ++k) tab.names.push_back("__o" + std::to_string(k));
    tab.types = keys.types;
    tab.names.push_back("__v");
    LType vt = LType::BIGINT;
    for (auto& row : keys.vals) {
      Stmt one = cloneStmt(sub);
      std::function<void(ExprP&)> bindOuter = [&](ExprP& x) {
        if (!x) return;
        if (x->kind == Expr::COL && outerRef(cat_, x, sub, s, q.ctes)) {
          const std::string key = lower(x->table + "." + x->name);
          size_t k = 0;
          while (k < outerCols.size() && lower(outerCols[k]->table + "." + outerCols[k]->name) != key) ++k;
          x = mkLit(row[k]);
          return;
        }
        for (auto& a : x->args) bindOuter(a);
      };
      walkStmtExprsDeep(one, bindOuter);
      ResultSet r = query(one, q, false, true);
      if (r.names.size() != 1) throw SqlError("Operand should contain 1 column(s)", 1241);
      if (r.vals.size() > 1) throw SqlError("Subquery returns more than 1 row", 1242);
      if (!r.types.empty()) vt = r.types[0];
      std::vector<Value> out = row;
      out.push_back(r.vals.empty() ? Value::Null() : r.vals[0][0]);
      tab.vals.push_back(std::move(out));
    }
    tab.types.push_back(vt);
    const std::string tmp = makeTemp(tab, q, "apply");
    static std::atomic<int> sq{0};
    const std::string alias = "__ap" + std::to_string(++sq);
    TableRef ref;
    ref.name = tmp;
    ref.alias = alias;
    ref.join = 1;
    // NULL outer values never match "=": with one outer column, its result is picked by IF(col IS
    // NULL, ..); with more, the columns holding NULLs are joined NULL-safely.
    auto isNull = [](ExprP x) {
      auto n = std::make_shared<Expr>();
      n->kind = Expr::FUNC;
      n->op = "ISNULL";
      n->args = {x};
      return n;
    };
    std::vector<bool> nullKey(outerCols.size());
    std::optional<Value> nullVal;
    for (auto& row : tab.vals)
      for (size_t k = 0; k < outerCols.size(); ++k)
        if (row[k].isNull()) {
          nullKey[k] = true;
          if (outerCols.size() == 1) nullVal = row.back();
        }
    for (size_t k = 0; k < outerCols.size(); ++k) {
      ExprP l = mkCol(alias, "__o" + std::to_string(k)), r = cloneExpr(outerCols[k]);
      ExprP eq = mkBin("=", l, r);
      if (nullKey[k] && outerCols.size() > 1) {
        auto co = std::make_shared<Expr>();
        co->kind = Expr::FUNC;
        co->op = "COALESCE";
        co->args = {eq, mkBin("AND", isNull(cloneExpr(l)), isNull(cloneExpr(r)))};
        eq = co;
      }
      ref.on = ref.on ? mkBin("AND", ref.on, eq) : eq;
    }
    s.from.push_back(ref);
    e = mkCol(alias, "__v");
    if (nullVal && !nullVal->isNull()) {
      auto f = std::make_shared<Expr>();
      f->kind = Expr::FUNC;
      f->op = "IF";
      f->args = {isNull(cloneExpr(outerCols[0])), mkLit(*nullVal), e};
      e = f;
    }
  };
  // Any other correlated EXISTS: (SELECT 1 .. LIMIT 1) IS NOT NULL, evaluated per outer value.
  auto existsPerOuterValue = [&](ExprP& e) {
    Stmt sub = cloneStmt(*e->sub);
    sub.items = {SelectItem{mkLit(Value::I(1)), ""}};
    sub.orderBy.clear();
    sub.limit = 1;
    sub.offset = 0;
    auto sq = std::make_shared<Expr>();
    sq->kind = Expr::SUBQUERY;
    sq->sub = std::make_shared<Stmt>(std::move(sub));
    ExprP v = sq;
    applyPerOuterValue(v);
    auto nn = std::make_shared<Expr>();
    nn->kind = Expr::FUNC;
    nn->op = "ISNOTNULL";
    nn->args = {v};
    e = nn;
  };
  std::function<void(ExprP&)> fix = [&](ExprP& e) {
    if (!e) return;
    for (auto& a : e->args) fix(a);
    if (e->kind == Expr::SUBQUERY) {
      Stmt sub = cloneStmt(*e->sub);
      if (correlated(sub)) {
        // (SELECT agg(x) FROM t2 WHERE t2.k = outer.k AND ..) -> LEFT JOIN (SELECT k, agg(x) .. GROUP BY k)
        std::vector<ExprP> conj, rest;
        std::function<void(const ExprP&)> flat = [&](const ExprP& x) {
          if (x && x->kind == Expr::BIN && x->op == "AND") { flat(x->args[0]); flat(x->args[1]); }
          else if (x) conj.push_back(x);
        };
        flat(sub.where);
        std::vector<ExprP> outerCols, innerCols;  // every inner.col = outer.col equality
        for (auto& c : conj) {
          if (c->kind == Expr::BIN && c->op == "=" && c->args[0]->kind == Expr::COL && c->args[1]->kind == Expr::COL) {
            bool o0 = outerRef(cat_, c->args[0], sub, s, q.ctes), o1 = outerRef(cat_, c->args[1], sub, s, q.ctes);
            if (o0 != o1) {
              outerCols.push_back(o0 ? c->args[0] : c->args[1]);
              innerCols.push_back(o0 ? c->args[1] : c->args[0]);
              continue;
            }
          }
          rest.push_back(c);
        }
        if (outerCols.empty() || sub.items.size() != 1 || !sub.groupBy.empty()) {
          applyPerOuterValue(e);
          return;
        }
        ExprP w;
        for (auto& r : rest) w = w ? mkBin("AND", w, r) : r;
        sub.where = w;
        ExprP item = sub.items[0].e;
        // Aggregates (GPU and host ones) in the item: each becomes a column of the grouped derived
        // table; the expression around them (COALESCE(COUNT(*), 0), ..) is evaluated after the LEFT
        // JOIN, so outer rows without matches see the empty-set values (COUNT 0, others NULL).
        std::vector<ExprP> aggs;
        std::function<bool(const ExprP&)> anyAgg = [&](const ExprP& x) {
          if (!x) return false;
          if (x->kind == Expr::FUNC && !x->window && (isAggName(x->op) || isHostAggName(x->op))) return true;
          for (auto& a : x->args)
            if (anyAgg(a)) return true;
          return false;
        };
        if (!anyAgg(item)) {  // plain value: one row per key expected; MAX keeps it single-valued
          auto m = std::make_shared<Expr>();
          m->kind = Expr::FUNC;
          m->op = "MAX";
          m->args = {item};
          item = m;
        }
        static std::atomic<int> sq{0};
        const std::string alias = "__sq" + std::to_string(++sq);
        std::function<ExprP(const ExprP&)> lift = [&](const ExprP& x) -> ExprP {
          if (x->kind == Expr::FUNC && !x->window && (isAggName(x->op) || isHostAggName(x->op))) {
            const std::string col = "__a" + std::to_string(aggs.size());
            aggs.push_back(x);
            ExprP v = mkCol(alias, col);
            if (x->op == "COUNT") {  // no matching rows: COUNT is 0, not NULL
              auto f = std::make_shared<Expr>();
              f->kind = Expr::FUNC;
              f->op = "COALESCE";
              f->args = {v, mkLit(Value::I(0))};
              v = f;
            }
            return v;
          }
          auto y = std::make_shared<Expr>(*x);
          for (auto& a : y->args) a = lift(a);
          return y;
        };
        ExprP outerExpr = lift(item);
        sub.items.clear();
        sub.groupBy.clear();
        for (size_t k = 0; k < innerCols.size(); ++k) {
          sub.items.push_back(SelectItem{innerCols[k], "__k" + std::to_string(k)});
          sub.groupBy.push_back(cloneExpr(innerCols[k]));
        }
        for (size_t k = 0; k < aggs.size(); ++k) sub.items.push_back(SelectItem{aggs[k], "__a" + std::to_string(k)});
        sub.orderBy.clear();
        sub.limit = -1;
        if (correlated(sub)) {
          applyPerOuterValue(e);
          return;
        }
        TableRef ref;
        ref.alias = alias;
        ref.sub = std::make_shared<Stmt>(std::move(sub));
        ref.join = 1;
        for (size_t k = 0; k < outerCols.size(); ++k) {
          ExprP eq = mkBin("=", mkCol(alias, "__k" + std::to_string(k)), cloneExpr(outerCols[k]));
          ref.on = ref.on ? mkBin("AND", ref.on, eq) : eq;
        }
        s.from.push_back(ref);
        e = outerExpr;
        return;
      }
      ResultSet r = query(sub, q, false, true);
      if (r.names.size() != 1) throw SqlError("Operand should contain 1 column(s)", 1241);
      if (r.vals.size() > 1) throw SqlError("Subquery returns more than 1 row", 1242);
      e = mkLit(r.vals.empty() ? Value::Null() : r.vals[0][0]);
      return;
    }
    if (e->kind == Expr::FUNC && e->op == "EXISTS") {
      Stmt sub = cloneStmt(*e->sub);
      if (correlated(sub)) {
        if (decorrelateExists(e)) {
          fix(e);
          return;
        } else {
          // EXISTS (.. WHERE <equalities with the outer query> AND inner.c <> outer.d AND <local filters>)
          //   = COUNT(.. equalities, filters, c IS NOT NULL) - COUNT(.. equalities, filters, c = d) > 0
          // (both equality-correlated counts, decorrelated into grouped LEFT JOINs).
          std::vector<ExprP> conj;
          std::function<void(const ExprP&)> flat = [&](const ExprP& x) {
            if (x && x->kind == Expr::BIN && x->op == "AND") { flat(x->args[0]); flat(x->args[1]); }
            else if (x) conj.push_back(x);
          };
          flat(sub.where);
          ExprP neq;
          std::vector<ExprP> keep;
          bool ok = true;
          for (auto& c : conj) {
            bool corr = false;
            std::function<void(const ExprP&)> chk = [&](const ExprP& x) {
              if (!x) return;
              if (x->kind == Expr::COL && outerRef(cat_, x, sub, s, q.ctes)) corr = true;
              for (auto& a : x->args) chk(a);
            };
            chk(c);
            const bool isEq = c->kind == Expr::BIN && c->op == "=" && c->args[0]->kind == Expr::COL && c->args[1]->kind == Expr::COL;
            if (corr && !isEq) {
              if (neq || !(c->kind == Expr::BIN && c->op == "<>" && c->args[0]->kind == Expr::COL && c->args[1]->kind == Expr::COL)) ok = false;
              else neq = c;
            } else {
              keep.push_back(c);
            }
          }
          if (!ok || !neq) {
            existsPerOuterValue(e);
            return;
          }
          // EXISTS (.. inner.c <> outer.d ..) <=> MIN(c) <> d OR MAX(c) <> d over the rows matching the
          // equalities and filters (if any c differs from d, the smallest or the largest does): one
          // grouped derived table on the equality keys, LEFT JOINed, instead of a correlated scan.
          const bool o0 = outerRef(cat_, neq->args[0], sub, s, q.ctes);
          const ExprP inner = o0 ? neq->args[1] : neq->args[0], outer = o0 ? neq->args[0] : neq->args[1];
          std::vector<ExprP> innerKeys, outerKeys, local;
          for (auto& k : keep) {
            const bool isEq = k->kind == Expr::BIN && k->op == "=" && k->args[0]->kind == Expr::COL && k->args[1]->kind == Expr::COL;
            const bool a0 = isEq && outerRef(cat_, k->args[0], sub, s, q.ctes), a1 = isEq && outerRef(cat_, k->args[1], sub, s, q.ctes);
            if (isEq && a0 != a1) {
              outerKeys.push_back(a0 ? k->args[0] : k->args[1]);
              innerKeys.push_back(a0 ? k->args[1] : k->args[0]);
            } else {
              local.push_back(k);
            }
          }
          if (innerKeys.empty()) {
            existsPerOuterValue(e);
            return;
          }
          Stmt ds = cloneStmt(sub);
          ExprP w;
          for (auto& l : local) w = w ? mkBin("AND", w, cloneExpr(l)) : cloneExpr(l);
          ds.where = w;
          ds.items.clear();
          ds.groupBy.clear();
          for (size_t k = 0; k < innerKeys.size(); ++k) {
            ds.items.push_back(SelectItem{cloneExpr(innerKeys[k]), "__k" + std::to_string(k)});
            ds.groupBy.push_back(cloneExpr(innerKeys[k]));
          }
          auto agg = [&](const char* fn) {
            auto f = std::make_shared<Expr>();
            f->kind = Expr::FUNC;
            f->op = fn;
            f->args = {cloneExpr(inner)};
            return f;
          };
          ds.items.push_back(SelectItem{agg("MIN"), "__mn"});
          ds.items.push_back(SelectItem{agg("MAX"), "__mx"});
          ds.orderBy.clear();
          ds.limit = -1;
          if (correlated(ds)) {
            existsPerOuterValue(e);
            return;
          }
          static std::atomic<int> ex{0};
          const std::string alias = "__ex" + std::to_string(++ex);
          TableRef ref;
          ref.alias = alias;
          ref.sub = std::make_shared<Stmt>(std::move(ds));
          ref.join = 1;
          for (size_t k = 0; k < outerKeys.size(); ++k) {
            ExprP eq = mkBin("=", mkCol(alias, "__k" + std::to_string(k)), cloneExpr(outerKeys[k]));
            ref.on = ref.on ? mkBin("AND", ref.on, eq) : eq;
          }
          s.from.push_back(ref);
          ExprP differs = mkBin("OR", mkBin("<>", mkCol(alias, "__mn"), cloneExpr(outer)), mkBin("<>", mkCol(alias, "__mx"), cloneExpr(outer)));
          auto co = std::make_shared<Expr>();
          co->kind = Expr::FUNC;
          co->op = "COALESCE";
          co->args = {differs, mkLit(Value::I(0))};
          e = co;
          return;
        }
      } else {
        sub.limit = 1;
        ResultSet r = query(sub, q, false, true);
        e = mkLit(Value::I(!r.vals.empty()));
        return;
      }
    }
    if (e->kind == Expr::INLIST && e->sub) {
      Stmt sub = cloneStmt(*e->sub);
      // x [NOT] IN (set): membership f; with a NULL in the set, "not found" is NULL (IF(f, 1, NULL)).
      auto setTest = [&](ExprP f, bool hasNull) {
        if (hasNull) {
          auto i = std::make_shared<Expr>();
          i->kind = Expr::FUNC;
          i->op = "IF";
          i->args = {f, mkLit(Value::I(1)), mkLit(Value::Null())};
          f = i;
        }
        if (!e->neg) return f;
        auto n = std::make_shared<Expr>();
        n->kind = Expr::UNARY;
        n->op = "NOT";
        n->args = {f};
        return ExprP(n);
      };
      if (correlated(sub)) {
        // x IN (SELECT c .. WHERE w) -> EXISTS (SELECT 1 .. WHERE w AND c = x); NOT IN ->
        // NOT EXISTS (.. AND (c = x OR c IS NULL OR x IS NULL)) (true only when nothing could match).
        bool agg = !sub.groupBy.empty() || sub.having || sub.items.size() != 1 || sub.limit >= 0 || sub.unionNext || sub.distinct;
        for (auto& it : sub.items) agg = agg || hasAgg(it.e);
        if (agg || e->args.size() != 1) throw SqlError("this correlated IN subquery is not supported", 1235);
        const ExprP c = sub.items[0].e;
        ExprP x = cloneExpr(e->args[0]);
        std::function<void(ExprP&)> qualify = [&](ExprP& y) {  // outer columns, named by their outer table
          if (!y) return;
          if (y->kind == Expr::COL && y->table.empty()) {
            for (auto& f : s.from) {
              Stmt one;
              one.from = {f};
              if (!f.sub && fromHasColumn(cat_, one, y->name)) {
                y->table = f.alias.empty() ? f.name : f.alias;
                break;
              }
            }
          }
          for (auto& a : y->args) qualify(a);
        };
        qualify(x);
        ExprP cond = mkBin("=", cloneExpr(c), cloneExpr(x));
        if (e->neg) {
          auto isNull = [](ExprP a) {
            auto n = std::make_shared<Expr>();
            n->kind = Expr::FUNC;
            n->op = "ISNULL";
            n->args = {a};
            return n;
          };
          cond = mkBin("OR", mkBin("OR", cond, isNull(cloneExpr(c))), isNull(cloneExpr(x)));
        }
        sub.where = sub.where ? mkBin("AND", sub.where, cond) : cond;
        auto ex = std::make_shared<Expr>();
        ex->kind = Expr::FUNC;
        ex->op = "EXISTS";
        ex->sub = std::make_shared<Stmt>(std::move(sub));
        ExprP r = ex;
        if (e->neg) {
          auto nt = std::make_shared<Expr>();
          nt->kind = Expr::UNARY;
          nt->op = "NOT";
          nt->args = {ex};
          r = nt;
        }
        e = r;
        fix(e);
        return;
      }
      // Integer values: the distinct values computed on the GPU stay there, as a bitmap the filter
      // tests (no host round trip), when their range is compact.
      std::shared_ptr<std::vector<double>> pre;  // else: the distinct numbers, if they came back to the host
      {
        bool plain = sub.items.size() == 1 && sub.groupBy.empty() && !sub.having && sub.limit < 0 && !sub.unionNext;
        for (auto& it : sub.items) plain = plain && !hasAgg(it.e) && !it.e->window;
        if (plain) {
          Stmt ds = cloneStmt(sub);
          ds.distinct = true;
          ds.orderBy.clear();
          tColumnar = true;
          ResultSet r;
          try {
            r = query(ds, q, false, true);
          } catch (...) {
            tColumnar = false;
            throw;
          }
          tColumnar = false;
          const bool intTy = !r.types.empty() && (r.types[0] == LType::INT || r.types[0] == LType::BIGINT || r.types[0] == LType::DATE);
          if (r.devCols.size() == 1 && intTy) {
            const auto& dc = r.devCols[0];
            const bool neg = e->neg;
            if (r.devRows == 0) { e = mkLit(Value::I(neg ? 1 : 0)); return; }
            if (!(dc->mn <= dc->mx)) {  // only NULL: never true; NULL unless x is .. (x IN (NULL) is NULL)
              e = setTest(mkLit(Value::I(0)), true);
              return;
            }
            const double span = dc->mx - dc->mn + 1;
            if (span <= (double)(1 << 28) && span <= 64.0 * (double)std::max<int64_t>(r.devRows, 1 << 16)) {
              auto b = std::make_shared<DevBits>();
              b->base = (int64_t)dc->mn;
              b->nbits = (int64_t)span;
              bool hasNull = false;
              b->p = gpuBitsFromColumn(dc->p, r.types[0] == LType::BIGINT ? 1 : 0, r.devRows, b->base, b->nbits, &hasNull);
              auto f = std::make_shared<Expr>();
              f->kind = Expr::FUNC;
              f->op = "#INBITS";
              f->args = {mkBin("-", e->args[0], mkLit(Value::I(b->base)))};
              f->devAux = b;
              e = setTest(f, hasNull);
              return;
            }
          }
          if (r.numCols.size() == 1 && !r.types.empty() && r.types[0] != LType::VARCHAR && r.devCols.empty())
            pre = std::make_shared<std::vector<double>>(std::move(r.numCols[0]));
        }
      }
      // Only the set of values matters: numbers come back as one raw array (no row objects).
      ResultSet r;
      if (pre) {
        r.numCol = pre;
      } else {
        tRawNumber = true;
        try {
          r = query(sub, q, false, true);
        } catch (...) {
          tRawNumber = false;
          throw;
        }
        tRawNumber = false;
      }
      if (r.numCol) {
        auto vals = r.numCol;
        const size_t before = vals->size();
        vals->erase(std::remove_if(vals->begin(), vals->end(), [](double v) { return v != v; }), vals->end());
        const bool neg = e->neg, hasNull = vals->size() != before;
        if (before == 0) { e = mkLit(Value::I(neg ? 1 : 0)); return; }
        if (vals->empty()) { e = setTest(mkLit(Value::I(0)), true); return; }
        // Sorted distinct values: integers in a compact range through a bitmap (no sort), else a sort.
        double lo = (*vals)[0], hi = lo;
        bool integral = true;
        for (double v : *vals) {
          lo = std::min(lo, v), hi = std::max(hi, v);
          integral = integral && v == std::floor(v);
        }
        if (integral && hi - lo < (double)(1 << 28)) {
          const int64_t base = (int64_t)lo, span = (int64_t)(hi - lo) + 1;
          std::vector<uint64_t> bits((size_t)(span + 63) / 64, 0);
          for (double v : *vals) {
            const int64_t k = (int64_t)v - base;
            bits[(size_t)k >> 6] |= 1ull << (k & 63);
          }
          vals->clear();
          for (size_t w = 0; w < bits.size(); ++w)
            for (uint64_t b = bits[w]; b; b &= b - 1) vals->push_back((double)(base + (int64_t)(w * 64 + __builtin_ctzll(b))));
        } else {
          std::sort(vals->begin(), vals->end());
          vals->erase(std::unique(vals->begin(), vals->end()), vals->end());
        }
        auto f = std::make_shared<Expr>();
        f->kind = Expr::FUNC;
        f->op = "#INSORTED";
        f->args = {e->args[0]};
        f->aux = vals;
        e = setTest(f, hasNull);
        return;
      }
      if (r.names.size() != 1) throw SqlError("Operand should contain 1 column(s)", 1241);
      const bool neg = e->neg;
      ExprP x = e->args[0];
      bool hasNull = false;
      for (auto& row : r.vals) hasNull = hasNull || row[0].isNull();
      if (r.vals.empty()) { e = mkLit(Value::I(neg ? 1 : 0)); return; }
      if (r.types[0] == LType::VARCHAR) {
        std::set<std::string> vs;
        for (auto& row : r.vals)
          if (!row[0].isNull()) vs.insert(row[0].toString());
        if (vs.empty()) { e = setTest(mkLit(Value::I(0)), true); return; }
        auto in = std::make_shared<Expr>();
        in->kind = Expr::INLIST;
        in->args = {x};
        for (auto& v : vs) in->args.push_back(mkLit(Value::S(v)));
        e = setTest(in, hasNull);
        return;
      }
      auto vals = std::make_shared<std::vector<double>>();
      for (auto& row : r.vals)
        if (!row[0].isNull()) vals->push_back(numOf(row[0]));
      if (vals->empty()) { e = setTest(mkLit(Value::I(0)), true); return; }
      std::sort(vals->begin(), vals->end());
      vals->erase(std::unique(vals->begin(), vals->end()), vals->end());
      auto f = std::make_shared<Expr>();
      f->kind = Expr::FUNC;
      f->op = "#INSORTED";
      f->args = {x};
      f->aux = vals;
      e = setTest(f, hasNull);
    }
  };
  walkStmtExprs(s, fix);
}


// SUM(x + c) / SUM(x - c) / SUM(c + x) -> SUM(x) +/- c * COUNT(x) (exact, NULLs included): many such
// aggregates over one column (ClickBench Q30) collapse to two.
static void rewriteSumOffsets(Stmt& s) {
  std::function<ExprP(const ExprP&)> rw = [&](const ExprP& e) -> ExprP {
    if (!e) return e;
    if (e->kind == Expr::FUNC && e->op == "SUM" && !e->window && !e->distinct && e->args.size() == 1 &&
        e->args[0]->kind == Expr::BIN && (e->args[0]->op == "+" || e->args[0]->op == "-")) {
      const ExprP& a = e->args[0]->args[0];
      const ExprP& b = e->args[0]->args[1];
      auto numLit = [](const ExprP& x) { return x->kind == Expr::LIT && (x->lit.kind == Value::INT || x->lit.kind == Value::DBL) && !x->lit.isNull(); };
      const bool cRight = numLit(b) && !numLit(a), cLeft = e->args[0]->op == "+" && numLit(a) && !numLit(b);
      if (cRight || cLeft) {
        const ExprP x = cRight ? a : b, c = cRight ? b : a;
        auto sum = std::make_shared<Expr>();
        sum->kind = Expr::FUNC;
        sum->op = "SUM";
        sum->args = {x};
        auto cnt = std::make_shared<Expr>();
        cnt->kind = Expr::FUNC;
        cnt->op = "COUNT";
        cnt->args = {x};
        return mkBin(e->args[0]->op, sum, mkBin("*", c, cnt));
      }
    }
    for (auto& a : e->args) a = rw(a);
    return e;
  };
  for (auto& it : s.items) {
    ExprP n = rw(it.e);
    if (n != it.e && it.alias.empty()) it.alias = it.e->str();
    it.e = n;
  }
  s.having = rw(s.having);
  for (auto& o : s.orderBy) o.e = rw(o.e);
}

// A derived table that only projects and filters one base table is merged into the outer query:
// no temporary table, and (on a cluster aggregator) the outer aggregation can run on the leaves.
void Engine::flattenDerived(Stmt& s) {
  while (s.from.size() == 1 && s.from[0].sub && s.joinConds.empty()) {
    Stmt& in = *s.from[0].sub;
    if (in.type != Stmt::SELECT || in.from.size() != 1 || in.from[0].sub || in.from[0].join || !in.joinConds.empty() ||
        !in.groupBy.empty() || in.having || in.distinct || in.keepDerived || in.limit >= 0 || in.unionNext || !in.ctes.empty() || in.items.empty() ||
        !cat_.find(in.from[0].name))
      return;
    std::function<bool(const ExprP&)> complex = [&](const ExprP& e) -> bool {
      if (!e) return false;
      if (e->kind == Expr::STAR || e->sub || e->kind == Expr::SUBQUERY || e->window || hasAgg(e)) return true;
      for (auto& a : e->args)
        if (complex(a)) return true;
      return false;
    };
    std::map<std::string, ExprP> byName;
    for (auto& it : in.items) {
      if (complex(it.e)) return;
      const std::string n = lower(!it.alias.empty() ? it.alias : it.e->kind == Expr::COL ? it.e->name : it.e->str());
      if (byName.count(n)) return;  // duplicate output names: leave it to the general path
      byName[n] = it.e;
    }
    std::set<std::string> outAliases;
    bool star = false;
    for (auto& it : s.items) {
      star = star || it.e->kind == Expr::STAR;
      if (!it.alias.empty()) outAliases.insert(lower(it.alias));
    }
    if (star) return;
    const std::string alias = s.from[0].alias;
    bool unknown = false;
    std::function<void(ExprP&, bool)> rw = [&](ExprP& e, bool orderLevel) {
      if (!e) return;
      if (e->kind == Expr::COL && (e->table.empty() || strcasecmp(e->table.c_str(), alias.c_str()) == 0)) {
        if (orderLevel && e->table.empty() && outAliases.count(lower(e->name))) return;  // a select-list alias
        auto it = byName.find(lower(e->name));
        if (it == byName.end()) { unknown = true; return; }
        e = cloneExpr(it->second);
        return;
      }
      if (e->sub) unknown = true;  // correlated references into the derived table: keep it
      for (auto& a : e->args) rw(a, false);
    };
    Stmt merged = cloneStmt(s);
    for (auto& it : merged.items) {
      if (it.alias.empty()) it.alias = it.e->kind == Expr::COL ? it.e->name : it.e->str();
      rw(it.e, false);
    }
    rw(merged.where, false);
    rw(merged.having, true);
    for (auto& g : merged.groupBy) rw(g, true);
    for (auto& o : merged.orderBy) rw(o.e, true);
    if (unknown) return;
    if (in.where) merged.where = merged.where ? mkBin("AND", cloneExpr(in.where), merged.where) : cloneExpr(in.where);
    merged.from = {in.from[0]};
    s = std::move(merged);
  }
}

// (a AND b AND x) OR (a AND b AND y) -> a AND b AND (x OR y): conditions every OR branch repeats
// (typically the join equality, TPC-H Q19) become plain conjuncts the planner can join on.
ExprP factorOr(const ExprP& e) {
  if (!e) return e;
  if (e->kind == Expr::BIN && e->op == "AND") {
    auto y = std::make_shared<Expr>(*e);
    y->args = {factorOr(e->args[0]), factorOr(e->args[1])};
    return y;
  }
  if (!(e->kind == Expr::BIN && e->op == "OR")) return e;
  std::vector<ExprP> branches;
  std::function<void(const ExprP&)> ors = [&](const ExprP& x) {
    if (x->kind == Expr::BIN && x->op == "OR") { ors(x->args[0]); ors(x->args[1]); }
    else branches.push_back(x);
  };
  ors(e);
  std::vector<std::vector<ExprP>> conj(branches.size());
  for (size_t i = 0; i < branches.size(); ++i) {
    std::function<void(const ExprP&)> ands = [&](const ExprP& x) {
      if (x->kind == Expr::BIN && x->op == "AND") { ands(x->args[0]); ands(x->args[1]); }
      else conj[i].push_back(x);
    };
    ands(branches[i]);
  }
  std::vector<ExprP> common;
  for (auto& c : conj[0]) {
    const std::string k = c->str();
    bool all = true;
    for (size_t i = 1; i < conj.size() && all; ++i)
      all = std::any_of(conj[i].begin(), conj[i].end(), [&](const ExprP& x) { return x->str() == k; });
    if (all && std::none_of(common.begin(), common.end(), [&](const ExprP& x) { return x->str() == k; })) common.push_back(c);
  }
  if (common.empty()) return e;
  ExprP rest;
  bool anyEmpty = false;
  for (auto& cs : conj) {
    ExprP b;
    for (auto& c : cs)
      if (std::none_of(common.begin(), common.end(), [&](const ExprP& x) { return x->str() == c->str(); })) b = b ? mkBin("AND", b, c) : c;
    if (!b) anyEmpty = true;
    else rest = rest ? mkBin("OR", rest, b) : b;
  }
  ExprP out;
  for (auto& c : common) out = out ? mkBin("AND", out, c) : c;
  if (!anyEmpty && rest) out = mkBin("AND", out, rest);  // (a branch that was only common terms: OR is true)
  return out;
}

// ROLLUP / CUBE / GROUPING SETS: one GROUP BY per set (keys outside the set are NULL there,
// GROUPING(k) is 1 for them), concatenated with UNION ALL; ORDER BY / LIMIT apply to the whole.
ResultSet Engine::groupingSetsQuery(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  for (auto& it : s.items)
    if (it.alias.empty()) it.alias = it.e->kind == Expr::COL ? it.e->name : it.e->str();
  std::vector<OrderItem> outerOrder;
  for (auto& o : s.orderBy) {
    int pos = -1;
    if (o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT) pos = (int)o.e->lit.i;
    for (size_t i = 0; i < s.items.size() && pos < 0; ++i)
      if ((o.e->kind == Expr::COL && o.e->table.empty() && strcasecmp(o.e->name.c_str(), s.items[i].alias.c_str()) == 0) ||
          o.e->str() == s.items[i].e->str())
        pos = (int)i + 1;
    if (pos < 0) throw SqlError("ORDER BY with ROLLUP / CUBE / GROUPING SETS must name a select-list column", 1235);
    outerOrder.push_back({mkLit(Value::I(pos)), o.desc});
  }
  auto sets = s.groupingSets;
  std::stable_sort(sets.begin(), sets.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
  std::shared_ptr<Stmt> head;
  Stmt* last = nullptr;
  for (auto& set : sets) {
    auto b = std::make_shared<Stmt>(cloneStmt(s));
    b->groupingSets.clear();
    b->orderBy.clear();
    b->limit = -1;
    b->offset = 0;
    b->groupBy.clear();
    std::vector<std::string> in, out;
    for (size_t i = 0; i < s.groupBy.size(); ++i) {
      const bool keep = std::find(set.begin(), set.end(), (int)i) != set.end();
      (keep ? in : out).push_back(s.groupBy[i]->str());
      if (keep) b->groupBy.push_back(cloneExpr(s.groupBy[i]));
    }
    std::function<void(ExprP&)> rw = [&](ExprP& e) {
      if (!e) return;
      if (e->kind == Expr::FUNC && upper(e->op) == "GROUPING" && e->args.size() == 1) {
        const std::string k = e->args[0]->str();
        e = mkLit(Value::I(std::find(out.begin(), out.end(), k) != out.end() ? 1 : 0));
        return;
      }
      if (e->kind == Expr::FUNC && (isAggName(e->op) || isHostAggName(e->op))) return;  // aggregates keep their arguments
      if (std::find(out.begin(), out.end(), e->str()) != out.end()) {
        e = mkLit(Value::Null());
        return;
      }
      for (auto& a : e->args) rw(a);
    };
    for (auto& it : b->items) rw(it.e);
    rw(b->having);
    if (!head) head = b;
    else {
      last->unionNext = b;
      last->unionAll = true;
    }
    last = b.get();
  }
  Stmt outer;
  outer.type = Stmt::SELECT;
  auto star = std::make_shared<Expr>();
  star->kind = Expr::STAR;
  outer.items = {SelectItem{star, ""}};
  TableRef ref;
  ref.alias = "__gs";
  ref.sub = head;
  outer.from = {ref};
  outer.orderBy = outerOrder;
  outer.limit = s.limit;
  outer.offset = s.offset;
  return query(outer, q, explain, typed);
}

std::shared_ptr<Stmt> tvfStatement(const std::string& name, const std::vector<ExprP>& args);  // proc.cpp
void Engine::expandTableFunctions(Stmt& s) {
  for (auto& f : s.from)
    if (f.tvf) {
      f.sub = tvfStatement(f.name, f.tvfArgs);
      f.tvf = false;
      f.tvfArgs.clear();
      f.name.clear();
    }
}

void Engine::resolveNaturalJoins(Stmt& s) {
  auto colsOf = [&](const TableRef& f) {
    std::vector<std::string> out;
    if (f.sub) {
      for (auto& it : f.sub->items) out.push_back(it.alias.empty() ? it.e->name : it.alias);
    } else if (auto t = cat_.find(f.name)) {
      for (auto& c : t->cols) out.push_back(c.name);
    }
    return out;
  };
  for (size_t i = 1; i < s.from.size(); ++i) {
    if (!s.from[i].natural) continue;
    s.from[i].natural = false;
    ExprP cond;
    for (auto& c : colsOf(s.from[i]))
      for (size_t k = 0; k < i; ++k) {
        auto ck = colsOf(s.from[k]);
        if (std::none_of(ck.begin(), ck.end(), [&](const std::string& x) { return strcasecmp(x.c_str(), c.c_str()) == 0; })) continue;
        ExprP eq = mkBin("=", mkCol(s.from[k].alias, c), mkCol(s.from[i].alias, c));
        cond = cond ? mkBin("AND", cond, eq) : eq;
        s.from[i].usingCols.push_back(c);
        break;
      }
    if (!cond) continue;  // no shared columns: a cross join
    if (s.from[i].join == 0) s.joinConds.push_back(cond);
    else s.from[i].on = cond;
  }
}

// [LEFT] JOIN LATERAL (SELECT a, b ..) x ON TRUE, where the subquery yields at most one row (an
// aggregate, LIMIT 1: what ORMs such as Drizzle generate for relations): every x.a becomes the
// correlated scalar subquery (SELECT a ..), evaluated once per distinct outer value.
static void rewriteLateral(Stmt& s) {
  for (size_t i = 0; i < s.from.size(); ++i) {
    TableRef& f = s.from[i];
    if (!f.lateral || !f.sub) continue;
    const bool onTrue = !f.on || (f.on->kind == Expr::LIT && !f.on->lit.isNull() && f.on->lit.num() != 0);
    if (!onTrue || f.join >= 2) throw SqlError("LATERAL derived tables are supported with ON TRUE (inner or LEFT JOIN)", 1235);
    std::map<std::string, ExprP> repl;
    for (auto& it : f.sub->items) {
      if (it.e->kind == Expr::STAR) throw SqlError("SELECT * in a LATERAL derived table is not supported", 1235);
      Stmt one = cloneStmt(*f.sub);
      one.items = {SelectItem{cloneExpr(it.e), it.alias}};
      auto q = std::make_shared<Expr>();
      q->kind = Expr::SUBQUERY;
      q->sub = std::make_shared<Stmt>(std::move(one));
      repl[lower(!it.alias.empty() ? it.alias : it.e->kind == Expr::COL ? it.e->name : it.e->str())] = q;
    }
    const std::string alias = lower(f.alias);
    std::function<void(ExprP&)> fix = [&](ExprP& e) {
      if (!e) return;
      if (e->kind == Expr::COL && lower(e->table) == alias) {
        auto it = repl.find(lower(e->name));
        if (it == repl.end()) throw SqlError("Unknown column '" + e->table + "." + e->name + "' in 'field list'", 1054);
        auto c = std::make_shared<Expr>(*it->second);
        c->sub = std::make_shared<Stmt>(cloneStmt(*it->second->sub));
        e = c;
        return;
      }
      for (auto& a : e->args) fix(a);
    };
    s.from.erase(s.from.begin() + i);
    --i;
    walkStmtExprs(s, fix);
  }
}

// A single-table query whose WHERE / GROUP BY / aggregates use expressions the GPU can't run (host
// functions over several columns, ..) runs again as
//   SELECT .. FROM (SELECT *, f(a, b) AS __e0, .. FROM t WHERE ..) AS t GROUP BY .. :
// the inner query filters and evaluates them on the host over the rows the GPU produces.
static bool hostRewrite(const Stmt& s, Stmt& out) {
  if (s.from.empty() || s.unionNext || !s.groupingSets.empty() || !s.ctes.empty()) return false;
  for (auto& f : s.from)
    if (f.tvf) return false;
  const bool multi = s.from.size() > 1 || !s.joinConds.empty();
  out = cloneStmt(s);
  const std::string alias = multi ? std::string("__hj") : s.from[0].alias.empty() ? s.from[0].name : s.from[0].alias;
  auto inner = std::make_shared<Stmt>();
  inner->type = Stmt::SELECT;
  inner->keepDerived = true;  // (merging it back would undo the rewrite)
  auto star = std::make_shared<Expr>();
  star->kind = Expr::STAR;
  if (!multi) inner->items.push_back(SelectItem{star, ""});
  inner->from = out.from;
  inner->joinConds = out.joinConds;
  out.joinConds.clear();
  std::map<std::string, std::string> done;
  std::function<bool(const ExprP&)> hasCol = [&](const ExprP& x) {
    if (!x) return false;
    if (x->kind == Expr::COL) return true;
    for (auto& a : x->args)
      if (hasCol(a)) return true;
    return false;
  };
  std::function<bool(const ExprP&)> hasColFunc = [&](const ExprP& x) {
    if (!x || x->kind == Expr::SUBQUERY) return false;
    if (x->kind == Expr::FUNC && !isAgg(x) && hasCol(x)) return true;
    for (auto& a : x->args)
      if (hasColFunc(a)) return true;
    return false;
  };
  // Joins can't filter on the host: conjuncts with functions of columns move to the outer query.
  ExprP outerWhere;
  if (multi && out.where) {
    std::vector<ExprP> conj;
    std::function<void(const ExprP&)> split = [&](const ExprP& x) {
      if (x->kind == Expr::BIN && x->op == "AND") { split(x->args[0]); split(x->args[1]); }
      else conj.push_back(x);
    };
    split(out.where);
    ExprP in;
    for (auto& x : conj) {
      ExprP& dst = hasColFunc(x) ? outerWhere : in;
      dst = dst ? mkBin("AND", dst, x) : x;
    }
    inner->where = in;
  } else if (!multi) {
    inner->where = out.where;
  }
  if (multi) {  // (inner join conditions may hold WHERE conjuncts too)
    std::vector<ExprP> keep;
    for (auto& j : inner->joinConds) {
      if (hasColFunc(j)) outerWhere = outerWhere ? mkBin("AND", outerWhere, j) : j;
      else keep.push_back(j);
    }
    inner->joinConds = keep;
  }
  out.where = outerWhere;
  std::function<void(ExprP&)> lift = [&](ExprP& x) {
    if (!x || x->kind == Expr::SUBQUERY) return;
    if (x->kind == Expr::FUNC && !isAgg(x) && !hasAgg(x) && hasCol(x)) {
      const std::string k = x->str();
      auto it = done.find(k);
      if (it == done.end()) {
        it = done.emplace(k, "__e" + std::to_string(done.size())).first;
        inner->items.push_back(SelectItem{x, it->second});
      }
      auto c = std::make_shared<Expr>();
      c->kind = Expr::COL;
      c->name = it->second;
      c->table = alias;
      x = c;
      return;
    }
    for (auto& a : x->args) lift(a);
  };
  for (auto& it : out.items) {
    const bool bare = it.alias.empty() && it.e && it.e->kind == Expr::FUNC && !isAgg(it.e);
    const std::string name = it.e ? it.e->str() : "";
    lift(it.e);
    if (bare) it.alias = name;  // keep the column header
  }
  for (auto& g : out.groupBy) lift(g);
  lift(out.having);
  lift(out.where);
  for (auto& o : out.orderBy) lift(o.e);
  if (done.empty() && !inner->where) return false;
  if (!multi) {  // only the columns the outer query still uses (not *: every column would be evaluated on the host)
    std::set<std::string> itemAliases, lifted, cols;
    for (auto& it : out.items)
      if (!it.alias.empty()) itemAliases.insert(lower(it.alias));
    for (auto& [k, v] : done) lifted.insert(lower(v));
    bool star = false;
    std::function<void(const ExprP&)> scan = [&](const ExprP& x) {
      if (!x || x->kind == Expr::SUBQUERY) return;
      if (x->kind == Expr::COL) {
        const std::string n = lower(x->name);
        if (!lifted.count(n) && !(x->table.empty() && itemAliases.count(n))) cols.insert(x->name);
        return;
      }
      for (auto& a : x->args) scan(a);
    };
    for (auto& it : out.items) {
      if (it.e && it.e->kind == Expr::STAR) star = true;
      scan(it.e);
    }
    for (auto& g : out.groupBy) scan(g);
    scan(out.having);
    scan(out.where);
    for (auto& o : out.orderBy) scan(o.e);
    if (!star) {
      inner->items.erase(inner->items.begin());  // (the *)
      std::set<std::string> seen;
      for (auto& n : cols) {
        if (!seen.insert(lower(n)).second) continue;
        auto c = std::make_shared<Expr>();
        c->kind = Expr::COL;
        c->name = n;
        inner->items.push_back(SelectItem{c, ""});
      }
      if (inner->items.empty()) {
        auto one = std::make_shared<Expr>();
        one->kind = Expr::LIT;
        one->lit = Value::I(1);
        inner->items.push_back(SelectItem{one, "__one"});
      }
    }
  }
  if (multi) {  // the join's columns the outer query still uses, each under a unique name
    std::map<std::string, std::string> cols;
    std::set<std::string> itemAliases;  // GROUP BY / ORDER BY / HAVING may name select items
    for (auto& it : out.items)
      if (!it.alias.empty()) itemAliases.insert(lower(it.alias));
    bool star = false;
    std::function<void(ExprP&)> mapCols = [&](ExprP& x) {
      if (!x || x->kind == Expr::SUBQUERY) return;
      if (x->kind == Expr::STAR) return;  // COUNT(*) (a SELECT * item is checked below)
      if (x->kind == Expr::COL && x->table.empty() && itemAliases.count(lower(x->name))) return;
      if (x->kind == Expr::COL && x->table != alias) {
        const std::string k = (x->table.empty() ? "" : x->table + ".") + x->name;
        auto it = cols.find(k);
        if (it == cols.end()) {
          it = cols.emplace(k, "__c" + std::to_string(cols.size())).first;
          auto src = std::make_shared<Expr>(*x);
          inner->items.push_back(SelectItem{src, it->second});
        }
        const std::string orig = x->name;
        auto c = std::make_shared<Expr>();
        c->kind = Expr::COL;
        c->name = it->second;
        c->table = alias;
        x = c;
        return;
      }
      for (auto& a : x->args) mapCols(a);
    };
    for (auto& it : out.items) {
      if (it.e && it.e->kind == Expr::STAR) star = true;
      if (it.e && it.e->kind == Expr::COL && it.alias.empty()) it.alias = it.e->name;  // keep the column header
      mapCols(it.e);
    }
    for (auto& g : out.groupBy) mapCols(g);
    mapCols(out.having);
    mapCols(out.where);
    for (auto& o : out.orderBy) mapCols(o.e);
    if (star) return false;
    if (inner->items.empty()) inner->items.push_back(SelectItem{std::make_shared<Expr>(Expr{}), ""}), inner->items.back().e->kind = Expr::LIT;
  }
  TableRef d;
  d.sub = inner;
  d.alias = alias;
  out.from = {d};
  return true;
}

ResultSet Engine::query(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  thread_local int tQueryDepth = 0;
  if (tQueryDepth == 0) cudaMemsetAsync(ovfFlagDev(), 0, sizeof(int), 0);
  ++tQueryDepth;
  struct QD { ~QD() { --tQueryDepth; } } qd;
  if (tQueryDepth == 1) {  // the outermost query: compiled int64 arithmetic that overflowed is an error (MySQL)
    ResultSet r = queryRetry(s, q, explain, typed);
    int ovf = 0;
    cudaMemcpy(&ovf, ovfFlagDev(), sizeof(int), cudaMemcpyDeviceToHost);
    if (ovf) throw SqlError("BIGINT value is out of range", 1690);
    return r;
  }
  return queryRetry(s, q, explain, typed);
}

ResultSet Engine::queryRetry(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  thread_local int tHostRewrite = 0;
  Stmt orig;
  const bool canRetry = !explain && tHostRewrite == 0 && !s.from.empty() && !s.unionNext;
  if (canRetry) orig = cloneStmt(s);
  try {
    return queryImpl(s, q, explain, typed);
  } catch (SqlError& e) {
    Stmt re;
    if (!canRetry || e.code != 1235 || !hostRewrite(orig, re)) throw;
    ++tHostRewrite;
    struct Dec { int& n; ~Dec() { --n; } } dec{tHostRewrite};
    return queryImpl(re, q, explain, typed);
  }
}

ResultSet Engine::queryImpl(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  // tRawNumber applies to this statement's own projection only, not to the queries run on the way.
  const bool rawNumber = tRawNumber && !s.unionNext && !explain, columnar = tColumnar && !s.unionNext && !explain;
  const bool insertCols = tInsertCols && !s.unionNext && !explain;
  tRawNumber = tColumnar = tInsertCols = false;
  struct RawRestore {
    bool v;
    ~RawRestore() { tRawNumber = tColumnar = tInsertCols = false; }
  } rawRestore{rawNumber};
  expandTableFunctions(s);
  if (!s.groupingSets.empty()) return groupingSetsQuery(s, q, explain, typed);
  rewriteLateral(s);
  resolveNaturalJoins(s);
  flattenDerived(s);
  s.where = factorOr(s.where);
  for (auto& j : s.joinConds) j = factorOr(j);
  rewriteSumOffsets(s);
  dropDependentKeys(s);
  auto savedCtes = q.ctes;
  struct Restore {
    QueryCtx& q;
    std::map<std::string, std::string> saved;
    ~Restore() { q.ctes = saved; }
  } restore{q, savedCtes};
  for (auto& [name, cs] : s.ctes) {
    // WITH RECURSIVE: anchor UNION [ALL] parts that reference the CTE itself, iterated to a fixpoint.
    bool recursive = false;
    for (Stmt* p = cs->unionNext.get(); p && !recursive; p = p->unionNext.get())
      for (auto& f : p->from) recursive |= !f.sub && strcasecmp(f.name.c_str(), name.c_str()) == 0;
    if (!recursive) {
      Stmt c = cloneStmt(*cs);
      ResultSet r = query(c, q, false, true);
      q.ctes[lower(name)] = makeTemp(r, q, name);
      continue;
    }
    Stmt anchor = cloneStmt(*cs);
    anchor.unionNext.reset();
    const bool all = cs->unionAll;
    ResultSet acc = query(anchor, q, false, true);
    std::set<std::string> seen;
    auto key = [](const std::vector<Value>& v) {
      std::string k;
      for (auto& x : v) k += (x.isNull() ? std::string("\x02") : x.toString()) + '\x1f';
      return k;
    };
    if (!all) {
      std::vector<std::vector<Value>> u;
      for (auto& v : acc.vals)
        if (seen.insert(key(v)).second) u.push_back(v);
      acc.vals.swap(u);
    }
    ResultSet delta = acc;
    for (int iter = 0; iter < 1000 && !delta.vals.empty(); ++iter) {
      q.ctes[lower(name)] = makeTemp(delta, q, name);
      Stmt rec = cloneStmt(*cs->unionNext);
      ResultSet r = query(rec, q, false, true);
      delta.vals.clear();
      for (auto& v : r.vals)
        if (all || seen.insert(key(v)).second) delta.vals.push_back(v);
      for (auto& v : delta.vals) acc.vals.push_back(v);
      if (iter == 999) throw SqlError("Recursive query aborted after 1000 iterations", 3636);
    }
    q.ctes[lower(name)] = makeTemp(acc, q, name);
  }
  if (s.unionNext) {
    std::vector<Stmt*> parts;
    for (Stmt* p = &s; p; p = p->unionNext.get()) parts.push_back(p);
    bool dedupe = false;
    size_t dedupeUpTo = 0;  // rows of parts [0, dedupeUpTo) are deduplicated (a UNION DISTINCT covers all to its left; UNION ALL after it doesn't)
    for (size_t i = 0; i + 1 < parts.size(); ++i)
      if (!parts[i]->unionAll || parts[i]->setOp != 0) dedupe = true, dedupeUpTo = i + 2;
    auto rowKey = [](const std::vector<Value>& v) {
      std::string k;
      for (auto& x : v) k += (x.isNull() ? std::string("\x02") : x.toString()) + '\x1f';
      return k;
    };
    ResultSet out;
    out.isResult = true;
    out.typed = true;
    bool allUnion = dedupe;  // UNION (no INTERSECT / EXCEPT): each part can drop its duplicates first, on the GPU
    for (size_t i = 0; i + 1 < parts.size(); ++i) allUnion = allUnion && parts[i]->setOp == 0 && !parts[i]->unionAll;
    std::vector<bool> typeKnown;
    size_t dedupeRows = 0;  // rows from the parts the deduplication covers
    for (size_t i = 0; i < parts.size(); ++i) {
      Stmt ps = cloneStmt(*parts[i]);
      ps.unionNext.reset();
      ps.ctes.clear();
      if (allUnion && ps.limit < 0 && !(i == 0 && s.unionOrder && !ps.orderBy.empty())) ps.distinct = true;
      if (i == 0 && s.unionOrder) {
        ps.orderBy.clear();
        ps.limit = -1;
        ps.offset = 0;
      }
      ResultSet r = query(ps, q, false, true);
      // A column that is NULL in every row of a part (a NULL literal, ROLLUP totals) takes the type of the others.
      auto allNull = [&](size_t c) {
        for (auto& v : r.vals)
          if (!v[c].isNull()) return false;
        return true;
      };
      if (i == 0) {
        out.names = r.names;
        out.types = r.types;
        typeKnown.assign(r.types.size(), false);
        for (size_t c = 0; c < r.types.size(); ++c) typeKnown[c] = !allNull(c);
      } else {
        if (r.names.size() != out.names.size())
          throw SqlError("The used SELECT statements have a different number of columns", 1222);
        for (size_t c = 0; c < out.types.size(); ++c) {
          LType a = out.types[c], b = r.types[c];
          if (a == b || allNull(c)) continue;
          if (!typeKnown[c]) {
            out.types[c] = b;
            typeKnown[c] = true;
            continue;
          }
          bool na = a == LType::INT || a == LType::BIGINT || a == LType::DOUBLE, nb = b == LType::INT || b == LType::BIGINT || b == LType::DOUBLE;
          out.types[c] = na && nb ? (a == LType::DOUBLE || b == LType::DOUBLE ? LType::DOUBLE : LType::BIGINT) : LType::VARCHAR;
        }
      }
      const int op = i == 0 ? 0 : parts[i - 1]->setOp;
      if (op == 0) {
        for (auto& v : r.vals) out.vals.push_back(std::move(v));
        if (i < dedupeUpTo) dedupeRows = out.vals.size();
      } else {  // INTERSECT / EXCEPT against the rows accumulated so far
        std::unordered_set<std::string> other;
        other.reserve(r.vals.size() * 2);
        for (auto& v : r.vals) other.insert(rowKey(v));
        std::vector<std::vector<Value>> keep;
        for (auto& v : out.vals)
          if ((other.count(rowKey(v)) > 0) == (op == 1)) keep.push_back(std::move(v));
        out.vals.swap(keep);
        if (i < dedupeUpTo) dedupeRows = out.vals.size();
      }
    }
    if (dedupe) {
      // keys built on all cores, then a hash set (the first of equal rows stays, in order)
      const size_t m = std::min(dedupeRows, out.vals.size());
      std::vector<std::string> keyOf(m);
      parallelFor(m, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) keyOf[i] = rowKey(out.vals[i]);
      });
      std::unordered_set<std::string_view> seen;
      seen.reserve(m * 2);
      std::vector<std::vector<Value>> keep;
      for (size_t i = 0; i < m; ++i)
        if (seen.insert(keyOf[i]).second) keep.push_back(std::move(out.vals[i]));
      for (size_t i = m; i < out.vals.size(); ++i) keep.push_back(std::move(out.vals[i]));  // (after the last UNION DISTINCT: kept)
      out.vals.swap(keep);
    }
    if (!s.orderBy.empty()) {
      std::vector<std::pair<int, bool>> keys;
      for (auto& o : s.orderBy) {
        int idx = -1;
        if (o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT) idx = (int)o.e->lit.i - 1;
        else if (o.e->kind == Expr::COL)
          for (size_t c = 0; c < out.names.size(); ++c)
            if (strcasecmp(out.names[c].c_str(), o.e->name.c_str()) == 0) idx = (int)c;
        if (idx < 0 || idx >= (int)out.names.size()) throw SqlError("Unknown column '" + o.e->str() + "' in 'order clause'", 1054);
        keys.push_back({idx, o.desc});
      }
      std::stable_sort(out.vals.begin(), out.vals.end(), [&](const std::vector<Value>& a, const std::vector<Value>& b) {
        for (auto& [i, d] : keys) {
          int c = cmpVal(a[i], b[i]);
          if (c) return d ? c > 0 : c < 0;
        }
        return false;
      });
    }
    if (s.offset > 0 || s.limit >= 0) {
      size_t st = std::min<size_t>(s.offset, out.vals.size());
      size_t en = s.limit < 0 ? out.vals.size() : std::min<size_t>(out.vals.size(), st + s.limit);
      out.vals = std::vector<std::vector<Value>>(out.vals.begin() + st, out.vals.begin() + en);
    }
    if (!typed) formatRows(out);
    return out;
  }
  // FROM: derived tables, CTE references, views and information_schema become temp tables.
  auto resolveFrom = [&]() {
  std::string viewDb;
  for (auto& f : s.from) {
    if (!f.sub && !q.ctes.count(lower(f.name))) {
      if (lower(f.name).rfind("information_schema.", 0) == 0 || (lower(currentDb()) == "information_schema" && f.name.find('.') == std::string::npos)) {
        ResultSet r = infoSchema(plainName(f.name));
        f.name = makeTemp(r, q, "is_" + plainName(f.name));
        continue;
      }
      std::string view;
      {
        std::lock_guard<std::mutex> g(cat_.mu);
        auto it = cat_.views.find(Catalog::key(f.name));
        if (it != cat_.views.end()) view = it->second, viewDb = it->first.substr(0, it->first.find('.'));
      }
      if (!view.empty()) {
        std::vector<std::string> names;
        if (view.rfind("/*cols:", 0) == 0) {
          size_t e = view.find("*/");
          std::string list = view.substr(7, e - 7);
          for (size_t a = 0, b; a <= list.size(); a = b + 1) {
            b = list.find(',', a);
            if (b == std::string::npos) b = list.size();
            names.push_back(list.substr(a, b - a));
          }
          view = view.substr(e + 2);
        }
        f.sub = std::make_shared<Stmt>(parseSql(view));
        for (size_t i = 0; i < names.size() && i < f.sub->items.size(); ++i) f.sub->items[i].alias = names[i];
      }
    }
    if (f.sub) {
      Stmt sub = cloneStmt(*f.sub);
      tColumnar = true;  // a numeric GROUP BY result may come back as columns (straight into the temp table)
      tInsertCols = true;  // ... and a projection too
      struct DbSwitch {  // a view's tables resolve in the view's database
        std::string saved;
        bool on;
        DbSwitch(const std::string& d) : saved(currentDb()), on(!d.empty()) { if (on) currentDb() = d; }
        ~DbSwitch() { if (on) currentDb() = saved; }
      } sw(viewDb);
      viewDb.clear();
      ResultSet r = query(sub, q, false, true);
      tColumnar = tInsertCols = false;
      f.name = makeTemp(r, q, f.alias);
      f.sub.reset();
    } else {
      auto it = q.ctes.find(lower(f.name));
      if (it != q.ctes.end()) f.name = it->second;
    }
  }
  };
  resolveFrom();
  resolveSubqueries(s, q);
  resolveFrom();  // derived tables added by subquery decorrelation
  {
    bool win = false;
    std::function<void(const ExprP&)> find = [&](const ExprP& e) {
      if (!e || win) return;
      if (e->window) win = true;
      for (auto& a : e->args) find(a);
    };
    for (auto& it : s.items) find(it.e);
    for (auto& o : s.orderBy) find(o.e);
    if (win) {
      tWinOuterCols = insertCols;  // (its final projection may come back as columns)
      return windows(s, q, explain, typed);
    }
  }
  {  // aggregates only the host computes (GROUP_CONCAT, JSON_ARRAYAGG, STDDEV, ..): host aggregation
    bool hostAgg = false;
    std::function<void(const ExprP&)> find = [&](const ExprP& e) {
      if (!e || hostAgg) return;
      if (e->kind == Expr::FUNC && isHostAggName(e->op) && !e->window) hostAgg = true;
      for (auto& a : e->args) find(a);
    };
    for (auto& it : s.items) find(it.e);
    find(s.having);
    for (auto& o : s.orderBy) find(o.e);
    if (hostAgg) return hostAggregate(s, q, explain, typed);
  }
  // DISTINCT aggregates (COUNT/SUM/AVG(DISTINCT x), mixed with ordinary aggregates): two-level
  // aggregation. The inner query groups by the GROUP BY keys plus x and computes partial aggregates;
  // the outer one aggregates x (now distinct per group) and combines the partials.
  std::vector<ExprP> dist;
  std::vector<ExprP> plainAggs;
  std::function<void(const ExprP&)> findAggs = [&](const ExprP& e) {
    if (!e) return;
    if (e->kind == Expr::FUNC && isAggName(e->op) && !e->window) {
      (e->distinct ? dist : plainAggs).push_back(e);
      return;
    }
    for (auto& a : e->args) findAggs(a);
  };
  for (auto& it : s.items) findAggs(it.e);
  findAggs(s.having);
  for (auto& o : s.orderBy) findAggs(o.e);
  bool gpuDistinct = !dist.empty() && s.from.size() == 1 && !s.from[0].sub && s.from[0].join == 0;
  for (auto& d : dist) gpuDistinct &= d->op == "COUNT" && d->args[0]->kind != Expr::STAR && d->args[0]->str() == dist[0]->args[0]->str();
  if (!dist.empty() && !gpuDistinct) {
    const std::string x = dist[0]->args[0]->str();
    bool mixed = false;
    for (auto& d : dist) mixed = mixed || d->args[0]->str() != x;
    if (mixed && s.groupBy.empty() && !s.having && !s.unionNext) {
      // Without GROUP BY, each DISTINCT aggregate is a scalar subquery over the same rows (each runs on
      // the GPU); with no other aggregate left, the outer query needs no rows at all.
      Stmt outer = cloneStmt(s);
      std::function<void(ExprP&)> rw = [&](ExprP& e) {
        if (!e) return;
        if (e->kind == Expr::FUNC && isAggName(e->op) && !e->window && e->distinct) {
          auto sub = std::make_shared<Stmt>();
          sub->type = Stmt::SELECT;
          sub->from = s.from;
          sub->joinConds = s.joinConds;
          sub->where = s.where ? cloneExpr(s.where) : nullptr;
          sub->items.push_back({cloneExpr(e), "v"});
          auto sq = std::make_shared<Expr>();
          sq->kind = Expr::SUBQUERY;
          sq->sub = sub;
          e = sq;
          return;
        }
        for (auto& a : e->args) rw(a);
      };
      for (auto& it : outer.items) {
        if (it.alias.empty()) it.alias = it.e->str();
        rw(it.e);
      }
      for (auto& o : outer.orderBy) rw(o.e);
      bool agg = false;
      for (auto& it : outer.items) agg = agg || hasAgg(it.e);
      if (!agg) {
        outer.from.clear();
        outer.joinConds.clear();
        outer.where.reset();
      }
      return queryImpl(outer, q, explain, typed);
    }
    if (mixed && !explain && !s.unionNext && s.groupingSets.empty()) {
      try {
        Stmt copy = cloneStmt(s);
        return multiDistinct(copy, q, typed);
      } catch (SqlError&) {
      }
    }
    if (mixed) return hostAggregate(s, q, explain, typed);  // mixed DISTINCT with GROUP BY: host path
    for (auto& d : dist)
      if (d->op == "MIN" || d->op == "MAX") d->distinct = false;  // MIN/MAX(DISTINCT x) = MIN/MAX(x)
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
    Stmt inner = cloneStmt(s);
    inner.items.clear();
    inner.groupBy.clear();
    for (size_t i = 0; i < groups.size(); ++i) {
      inner.items.push_back({cloneExpr(groups[i]), "__g" + std::to_string(i)});
      inner.groupBy.push_back(cloneExpr(groups[i]));
    }
    inner.items.push_back({cloneExpr(dist[0]->args[0]), "__d"});
    inner.groupBy.push_back(cloneExpr(dist[0]->args[0]));
    // Partial aggregates of the ordinary aggregates, per (keys, x).
    std::map<std::string, ExprP> outerOf;  // canonical text of an ordinary aggregate -> its outer form
    auto mkAgg = [](const std::string& fn, ExprP arg) {
      auto f = std::make_shared<Expr>();
      f->kind = Expr::FUNC;
      f->op = fn;
      f->args = {arg};
      return f;
    };
    int np = 0;
    for (auto& a : plainAggs) {
      const std::string k = a->str();
      if (outerOf.count(k)) continue;
      auto part = [&](const std::string& fn, const ExprP& arg) {
        std::string name = "__p" + std::to_string(np++);
        inner.items.push_back({mkAgg(fn, arg->kind == Expr::STAR ? arg : cloneExpr(arg)), name});
        return mkCol("", name);
      };
      ExprP o;
      if (a->op == "COUNT") {
        auto c = std::make_shared<Expr>();
        c->kind = Expr::FUNC;
        c->op = "COALESCE";
        c->args = {mkAgg("SUM", part("COUNT", a->args[0])), mkLit(Value::I(0))};
        o = c;
      } else if (a->op == "SUM") o = mkAgg("SUM", part("SUM", a->args[0]));
      else if (a->op == "MIN") o = mkAgg("MIN", part("MIN", a->args[0]));
      else if (a->op == "MAX") o = mkAgg("MAX", part("MAX", a->args[0]));
      else {  // AVG = SUM / COUNT
        ExprP sm = mkAgg("SUM", part("SUM", a->args[0])), cn = mkAgg("SUM", part("COUNT", a->args[0]));
        o = mkBin("/", sm, cn);
      }
      outerOf[k] = o;
    }
    inner.having.reset();
    inner.orderBy.clear();
    inner.limit = -1;
    inner.offset = 0;
    inner.distinct = false;
    tColumnar = true;  // numeric / dictionary-code GROUP BY results come back as columns
    ResultSet r;
    try {
      r = query(inner, q, false, true);
    } catch (...) {
      tColumnar = false;
      throw;
    }
    tColumnar = false;
    std::string tmp = makeTemp(r, q, "dist");
    std::function<void(ExprP&)> rw = [&](ExprP& e) {
      if (!e) return;
      for (size_t i = 0; i < groups.size(); ++i)
        if (e->str() == groups[i]->str()) { e = mkCol("", "__g" + std::to_string(i)); return; }
      if (e->kind == Expr::FUNC && isAggName(e->op)) {
        if (e->distinct) {
          auto c = std::make_shared<Expr>(*e);
          c->distinct = false;
          c->args = {mkCol("", "__d")};
          e = c;
        } else {
          e = cloneExpr(outerOf.at(e->str()));
        }
        return;
      }
      for (auto& a : e->args) rw(a);
    };
    for (auto& it : s.items) {
      if (it.alias.empty()) it.alias = it.e->str();
      rw(it.e);
    }
    rw(s.having);
    for (auto& o : s.orderBy) rw(o.e);
    s.groupBy.clear();
    for (size_t i = 0; i < groups.size(); ++i) s.groupBy.push_back(mkCol("", "__g" + std::to_string(i)));
    s.from = {TableRef{tmp, "__dist", nullptr, 0, nullptr}};
    s.joinConds.clear();
    s.where.reset();
  }
  bool savedWant = tWantVals;
  tWantVals = typed;
  ResultSet out;
  std::unique_ptr<Stmt> pristine;
  if (!s.groupBy.empty() || s.distinct) pristine = std::make_unique<Stmt>(cloneStmt(s));
  try {
    tRawNumber = rawNumber;
    tColumnar = columnar;
    tInsertCols = insertCols;
    out = doSelect(s, explain);
    tRawNumber = tColumnar = tInsertCols = false;
  } catch (SqlError& e) {
    tWantVals = savedWant;
    const std::string msg = e.what();
    // GROUP BY the GPU can't run (a string key it can't group, more than 8 keys): on the host.
    if (pristine && (msg.find("GROUP BY on this string") != std::string::npos || msg.find("GROUP BY expressions are supported") != std::string::npos)) {
      if (pristine->groupBy.empty() && pristine->distinct) {  // SELECT DISTINCT ..: GROUP BY its items
        bool star = false;
        for (auto& it : pristine->items) star = star || it.e->kind == Expr::STAR;
        if (star) {  // (DISTINCT *: every column of the one table)
          if (pristine->from.size() != 1 || pristine->from[0].sub || !pristine->joinConds.empty()) throw;
          auto t = cat_.get(pristine->from[0].name);
          pristine->items.clear();
          for (auto& c : t->cols) {
            auto col = std::make_shared<Expr>();
            col->kind = Expr::COL;
            col->name = c.name;
            pristine->items.push_back({col, ""});
          }
        }
        for (auto& it : pristine->items)
          if (hasAgg(it.e)) throw;
        for (auto& it : pristine->items) pristine->groupBy.push_back(cloneExpr(it.e));
        pristine->distinct = false;
      }
      if (msg.find("GROUP BY expressions are supported") != std::string::npos) {
        Stmt packed = cloneStmt(*pristine);
        if (packGroupKeys(packed)) {  // integer keys packed into one: on the GPU after all
          tWantVals = typed;
          try {
            out = doSelect(packed, explain);
            tWantVals = savedWant;
            return out;
          } catch (SqlError&) {
            tWantVals = savedWant;
          }
        }
      }
      return hostAggregate(*pristine, q, explain, typed);
    }
    throw;
  } catch (...) {
    tWantVals = savedWant;
    throw;
  }
  tWantVals = savedWant;
  if (!typed) formatRows(out);
  return out;
}

void Engine::expandStars(Stmt& s) {
  std::vector<SelectItem> items;
  for (auto& it : s.items) {
    if (it.e->kind != Expr::STAR) {
      items.push_back(it);
      continue;
    }
    for (auto& f : s.from) {
      if (!it.e->table.empty() && strcasecmp(it.e->table.c_str(), f.alias.c_str()) != 0) continue;
      auto t = cat_.get(f.name);
      for (auto& c : t->cols) items.push_back({mkCol(f.alias, c.name), c.name});
    }
  }
  s.items = std::move(items);
}

// Window functions: run the query without them (keeping every expression they need as hidden
// columns), compute the windows over the result on the host, then run the final projection,
// ORDER BY and LIMIT over a temp table with the results.
ResultSet Engine::windows(Stmt& s, QueryCtx& q, bool explain, bool typed) {
  const bool outerCols = tWinOuterCols;
  tWinOuterCols = false;
  expandStars(s);
  Stmt inner = cloneStmt(s);
  inner.items.clear();
  inner.orderBy.clear();
  inner.limit = -1;
  inner.offset = 0;
  inner.distinct = false;
  std::map<std::string, int> hiddenOf;
  auto hidden = [&](const ExprP& e) -> int {
    std::string k = e->str();
    auto it = hiddenOf.find(k);
    if (it != hiddenOf.end()) return it->second;
    int i = (int)inner.items.size();
    inner.items.push_back({cloneExpr(e), "__c" + std::to_string(i)});
    hiddenOf[k] = i;
    return i;
  };
  struct Win {
    ExprP node;
    std::vector<int> args, part;
    std::vector<std::pair<int, bool>> ord;
  };
  std::vector<Win> wins;
  std::map<const Expr*, int> winOf;
  std::function<ExprP(const ExprP&)> outerOf = [&](const ExprP& e) -> ExprP {
    if (e->window) {
      auto it = winOf.find(e.get());
      int w;
      if (it != winOf.end()) {
        w = it->second;
      } else {
        Win wn;
        wn.node = e;
        for (auto& a0 : e->args) {
          if (a0->kind == Expr::STAR) continue;
          // (MIN / MAX over a string bound before: its argument became the string's rank; the
          // window needs the strings themselves)
          const ExprP a = (e->op == "MIN" || e->op == "MAX") && a0->kind == Expr::FUNC && a0->op == "#MAP" && a0->args.size() == 1 ? a0->args[0] : a0;
          wn.args.push_back(hidden(a));
        }
        for (auto& p : e->partBy) wn.part.push_back(hidden(p));
        for (auto& [o, d] : e->winOrder) wn.ord.push_back({hidden(o), d});
        w = (int)wins.size();
        wins.push_back(wn);
        winOf[e.get()] = w;
      }
      return mkCol("", "__w" + std::to_string(w));
    }
    if (e->kind == Expr::COL || (e->kind == Expr::FUNC && isAggName(e->op))) return mkCol("", "__c" + std::to_string(hidden(e)));
    for (auto& g : inner.groupBy)  // a GROUP BY expression: as a whole (its columns alone are not grouped)
      if (g->str() == e->str()) return mkCol("", "__c" + std::to_string(hidden(e)));
    auto y = std::make_shared<Expr>(*e);
    y->bound = false;
    for (auto& a : y->args) a = outerOf(a);
    return y;
  };
  Stmt outer;
  outer.type = Stmt::SELECT;
  for (auto& it : s.items) outer.items.push_back({outerOf(it.e), it.alias.empty() ? (it.e->kind == Expr::COL ? it.e->name : it.e->str()) : it.alias});
  for (auto& o : s.orderBy) {
    bool alias = o.e->kind == Expr::COL && o.e->table.empty() &&
                 std::any_of(s.items.begin(), s.items.end(), [&](const SelectItem& it) { return !it.alias.empty() && strcasecmp(it.alias.c_str(), o.e->name.c_str()) == 0; });
    bool pos = o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT;
    outer.orderBy.push_back({alias || pos ? o.e : outerOf(o.e), o.desc});
  }
  outer.limit = s.limit;
  outer.offset = s.offset;
  outer.distinct = s.distinct;
  if (inner.items.empty()) inner.items.push_back({mkLit(Value::I(1)), "__c0"});
  const auto tW0 = Clock::now();
  Stmt innerRows = cloneStmt(inner);  // (in case the result comes back as numbers only: run again for rows)
  tColumnar = true;  // a GROUP BY inner query may come back built on the GPU (exact, typed columns)
  tInsertCols = true;  // ... and a projection as typed columns
  ResultSet r = query(inner, q, false, true);
  tColumnar = tInsertCols = false;
  if (r.devCols.empty() && !r.insertCols && (!r.numCols.empty() || r.numCol)) r = query(innerRows, q, false, true);
  if ((!r.devCols.empty() || r.insertCols) && !explain) {
    // Columnar: the inner result as a temp table; keys, sort and window values from its typed
    // columns (64-bit integer keys, strings by collation rank); the output appended a column at a time.
    const std::string srcName = makeTemp(r, q, "winsrc");
    auto src = cat_.get(srcName);
    const size_t nin = src->cols.size();
    const size_t n = src->segs.empty() ? 0 : (size_t)src->segs[0]->n;
    std::vector<const uint8_t*> raw(nin, nullptr);
    if (n)
      for (size_t c = 0; c < nin; ++c) raw[c] = src->colData(Loc{0, 0}, (int)c).data();
    auto isF = [&](int c) { return src->cols[c].pt == PType::F64; };
    auto rawI = [&](int c, size_t i) -> int64_t {
      return src->cols[c].pt == PType::I32 ? (int64_t)((const int32_t*)raw[c])[i] : ((const int64_t*)raw[c])[i];
    };
    auto rawD = [&](int c, size_t i) { return ((const double*)raw[c])[i]; };
    auto isNullAt = [&](int c, size_t i) {
      if (isF(c)) return rawD(c, i) != rawD(c, i);
      return rawI(c, i) == (src->cols[c].pt == PType::I32 ? (int64_t)kNullI32 : kNullI64);
    };
    struct KeyCol {
      bool dbl = false;
      std::vector<int64_t> i;
      std::vector<double> d;
    };
    std::map<int, KeyCol> keys;
    auto keyCol = [&](int col) -> const KeyCol& {
      auto it = keys.find(col);
      if (it != keys.end()) return it->second;
      KeyCol k;
      const ColumnMeta& cm = src->cols[col];
      std::shared_ptr<std::vector<double>> rank;
      if (cm.lt == LType::VARCHAR) rank = sortedDict(*src->dicts[col]).second;
      if (isF(col)) {
        k.dbl = true;
        k.d.resize(n);
        parallelFor(n, [&](size_t b, size_t e) {
          for (size_t i = b; i < e; ++i) k.d[i] = isNullAt(col, i) ? -HUGE_VAL : rawD(col, i);
        });
      } else {
        k.i.resize(n);
        parallelFor(n, [&](size_t b, size_t e) {
          for (size_t i = b; i < e; ++i) {
            if (isNullAt(col, i)) k.i[i] = INT64_MIN;
            else k.i[i] = rank ? (int64_t)(*rank)[(size_t)rawI(col, i)] : rawI(col, i);
          }
        });
      }
      return keys[col] = std::move(k);
    };
    std::vector<std::vector<int64_t>> outI;   // per window: integer / code results
    std::vector<std::vector<double>> outD;    // per window: double results
    std::vector<ColumnMeta> outMeta;
    std::vector<std::shared_ptr<Dict>> outDict;
    std::vector<size_t> lastOrderC;
    for (size_t w = 0; w < wins.size(); ++w) {
      const Win& wn = wins[w];
      const std::string f = wn.node->op;
      static const std::set<std::string> kFns = {"ROW_NUMBER", "RANK", "DENSE_RANK", "PERCENT_RANK", "CUME_DIST", "NTILE", "LAG", "LEAD",
                                                 "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "SUM", "AVG", "COUNT", "MIN", "MAX"};
      if (!kFns.count(f)) throw SqlError("window function " + f + " is not supported", 1235);
      std::vector<const KeyCol*> pk;
      std::vector<std::pair<const KeyCol*, bool>> ok;
      for (int p : wn.part) pk.push_back(&keyCol(p));
      for (auto& [o, d] : wn.ord) ok.push_back({&keyCol(o), d});
      auto cmpK = [](const KeyCol& k, size_t a, size_t b) { return k.dbl ? (k.d[a] < k.d[b] ? -1 : k.d[a] > k.d[b]) : (k.i[a] < k.i[b] ? -1 : k.i[a] > k.i[b]); };
      auto cmpKeys = [&](size_t a, size_t b, bool withOrder) {
        for (auto* k : pk)
          if (int c = cmpK(*k, a, b)) return c;
        if (withOrder)
          for (auto& [k, d] : ok)
            if (int c = cmpK(*k, a, b)) return d ? -c : c;
        return 0;
      };
      std::vector<size_t> idx(n);
      for (size_t i = 0; i < n; ++i) idx[i] = i;
      {  // stable sort on all cores: sorted chunks, then pairwise merges
        auto less = [&](size_t a, size_t b) { return cmpKeys(a, b, true) < 0; };
        const size_t nth = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), n / 32768));
        std::vector<size_t> bounds;
        for (size_t t = 0; t <= nth; ++t) bounds.push_back(n * t / nth);
        parallelFor(nth, [&](size_t b, size_t e) {
          for (size_t t = b; t < e; ++t) std::stable_sort(idx.begin() + bounds[t], idx.begin() + bounds[t + 1], less);
        }, 1);
        for (size_t step = 1; step < nth; step *= 2) {
          std::vector<size_t> starts;
          for (size_t t = 0; t + step < nth; t += 2 * step) starts.push_back(t);
          parallelFor(starts.size(), [&](size_t b, size_t e) {
            for (size_t k = b; k < e; ++k) {
              size_t t = starts[k];
              std::inplace_merge(idx.begin() + bounds[t], idx.begin() + bounds[t + step], idx.begin() + bounds[std::min(nth, t + 2 * step)], less);
            }
          }, 1);
        }
      }
      // Result type: ranks and counts BIGINT, ratios / AVG DOUBLE, SUM by its argument, value
      // functions the argument's own type (strings: codes of its dictionary).
      const int a0 = wn.args.empty() ? -1 : wn.args[0];
      ColumnMeta om;
      om.name = "__w" + std::to_string(w);
      om.lt = LType::BIGINT;
      std::shared_ptr<Dict> od;
      const bool decArg = a0 >= 0 && src->cols[a0].lt == LType::DECIMAL;
      const double argUnit = decArg ? pow10i(src->cols[a0].scale) : 1.0;  // (DECIMAL cells are scaled integers)
      if (f == "PERCENT_RANK" || f == "CUME_DIST" || f == "AVG") om.lt = LType::DOUBLE;
      else if (f == "SUM") {
        om.lt = a0 >= 0 && isF(a0) ? LType::DOUBLE : decArg ? LType::DECIMAL : LType::BIGINT;
        if (decArg) om.scale = src->cols[a0].scale;
      } else if ((f == "MIN" || f == "MAX" || f == "LAG" || f == "LEAD" || f == "FIRST_VALUE" || f == "LAST_VALUE" || f == "NTH_VALUE") && a0 >= 0) {
        om.lt = src->cols[a0].lt;
        om.scale = src->cols[a0].scale;
        if (om.lt == LType::VARCHAR) od = src->dicts[a0];
      }
      om.pt = physOf(om.lt);
      const bool fOut = om.pt == PType::F64;
      std::vector<int64_t> oi(fOut ? 0 : n);
      std::vector<double> od2(fOut ? n : 0);
      const int64_t onul = om.pt == PType::I32 ? (int64_t)kNullI32 : kNullI64;
      auto constArg = [&](size_t k, double def) {
        if (k >= wn.node->args.size()) return def;
        const ExprP& a = wn.node->args[k];
        return a->kind == Expr::LIT ? numOf(a->lit) : def;
      };
      // a cell of the argument column into the output (same physical type, or converted)
      auto putArg = [&](size_t row, int col, size_t at) {
        if (col < 0 || isNullAt(col, at)) {
          if (fOut) od2[row] = nullv();
          else oi[row] = onul;
          return;
        }
        if (fOut) od2[row] = isF(col) ? rawD(col, at) : (double)rawI(col, at);
        else oi[row] = isF(col) ? (int64_t)std::llround(rawD(col, at)) : rawI(col, at);
      };
      const KeyCol* valKey = a0 >= 0 && (f == "MIN" || f == "MAX") ? &keyCol(a0) : nullptr;  // (strings: by rank)
      std::vector<std::pair<size_t, size_t>> parts;
      for (size_t a = 0; a < n;) {
        size_t b = a + 1;
        while (b < n && cmpKeys(idx[a], idx[b], false) == 0) ++b;
        parts.push_back({a, b});
        a = b;
      }
      parallelFor(parts.size(), [&](size_t pb, size_t pe) {
        for (size_t pi = pb; pi < pe; ++pi) {
          const size_t a = parts[pi].first, b = parts[pi].second, cnt = b - a;
          std::vector<size_t> peerStart(cnt), peerEnd(cnt);
          for (size_t i = 0; i < cnt;) {
            size_t j = i + 1;
            while (j < cnt && cmpKeys(idx[a + i], idx[a + j], true) == 0) ++j;
            for (size_t k = i; k < j; ++k) peerStart[k] = i, peerEnd[k] = j;
            i = j;
          }
          // running aggregates over frames that start at the partition (prefix sums), or per frame
          double pAcc = 0;
          int64_t pAccI = 0, pCnt = 0;
          size_t pBest = SIZE_MAX, pUpTo = 0;
          int64_t dense = 0;
          for (size_t i = 0; i < cnt; ++i) {
            const size_t row = idx[a + i];
            if (i == 0 || peerStart[i] == i) ++dense;
            size_t lo, hi;
            if (wn.node->rowsFrame) {
              int64_t L = wn.node->frameLo == INT64_MIN ? 0 : std::max<int64_t>(0, (int64_t)i + wn.node->frameLo);
              int64_t H = wn.node->frameHi == INT64_MAX ? (int64_t)cnt : std::min<int64_t>((int64_t)cnt, (int64_t)i + wn.node->frameHi + 1);
              lo = (size_t)L;
              hi = (size_t)std::max<int64_t>(L, H);
            } else if (wn.ord.empty()) {
              lo = 0, hi = cnt;
            } else {
              lo = 0, hi = wn.node->frameHi == INT64_MAX ? cnt : peerEnd[i];
            }
            if (f == "ROW_NUMBER") oi[row] = (int64_t)i + 1;
            else if (f == "RANK") oi[row] = (int64_t)peerStart[i] + 1;
            else if (f == "DENSE_RANK") oi[row] = dense;
            else if (f == "PERCENT_RANK") od2[row] = cnt > 1 ? (double)peerStart[i] / (double)(cnt - 1) : 0.0;
            else if (f == "CUME_DIST") od2[row] = (double)peerEnd[i] / (double)cnt;
            else if (f == "NTILE") {
              int64_t k = std::max<int64_t>(1, (int64_t)constArg(0, 1)), sz = (int64_t)cnt / k, rem = (int64_t)cnt % k;
              int64_t pos = (int64_t)i, bucket = pos < rem * (sz + 1) ? pos / (sz + 1) : rem + (pos - rem * (sz + 1)) / std::max<int64_t>(sz, 1);
              oi[row] = bucket + 1;
            } else if (f == "LAG" || f == "LEAD") {
              int64_t off = (int64_t)constArg(1, 1), j = f == "LAG" ? (int64_t)i - off : (int64_t)i + off;
              if (j >= 0 && j < (int64_t)cnt) putArg(row, a0, idx[a + j]);
              else putArg(row, wn.args.size() > 2 ? wn.args[2] : -1, row);
            } else if (f == "FIRST_VALUE") {
              if (hi > lo) putArg(row, a0, idx[a + lo]);
              else putArg(row, -1, row);
            } else if (f == "LAST_VALUE") {
              if (hi > lo) putArg(row, a0, idx[a + hi - 1]);
              else putArg(row, -1, row);
            } else if (f == "NTH_VALUE") {
              size_t k = (size_t)std::max<double>(1, constArg(1, 1));
              if (lo + k - 1 < hi) putArg(row, a0, idx[a + lo + k - 1]);
              else putArg(row, -1, row);
            } else {  // SUM / AVG / COUNT / MIN / MAX over [lo, hi)
              const bool star = a0 < 0;
              double acc = 0;
              int64_t accI = 0, c = 0;
              size_t best = SIZE_MAX;
              auto take = [&](size_t at, double& sacc, int64_t& sI, int64_t& sc, size_t& sbest) {
                if (star) { ++sc; return; }
                if (isNullAt(a0, at)) return;
                ++sc;
                if (isF(a0)) sacc += rawD(a0, at);
                else sI += rawI(a0, at), sacc += (double)rawI(a0, at);
                if (valKey) {
                  const int cmp = sbest == SIZE_MAX ? 1 : cmpK(*valKey, at, sbest);
                  if (sbest == SIZE_MAX || (f == "MIN" ? cmp < 0 : cmp > 0)) sbest = at;
                }
              };
              if (lo == 0) {  // a prefix of the partition: extend the running totals
                for (; pUpTo < hi; ++pUpTo) take(idx[a + pUpTo], pAcc, pAccI, pCnt, pBest);
                acc = pAcc, accI = pAccI, c = pCnt, best = pBest;
              } else {
                for (size_t j = lo; j < hi; ++j) take(idx[a + j], acc, accI, c, best);
              }
              if (f == "COUNT") oi[row] = c;
              else if (c == 0) putArg(row, -1, row);
              else if (f == "SUM") {
                if (fOut) od2[row] = acc;
                else oi[row] = accI;  // (DECIMAL: the scaled sum, same scale)
              } else if (f == "AVG") od2[row] = acc / (double)c / argUnit;
              else putArg(row, a0, best);
            }
          }
        }
      }, std::max<size_t>(1, parts.size() / (4 * std::thread::hardware_concurrency())));
      if (w + 1 == wins.size() && s.orderBy.empty() && !s.distinct && (!wn.part.empty() || !wn.ord.empty())) lastOrderC = idx;
      outI.push_back(std::move(oi));
      outD.push_back(std::move(od2));
      outMeta.push_back(om);
      outDict.push_back(od);
    }
    // The output table: the inner columns, the window values (and the window order, without ORDER BY).
    std::vector<ColumnMeta> cols = src->cols;
    for (auto& m : outMeta) cols.push_back(m);
    if (!lastOrderC.empty()) {
      ColumnMeta m;
      m.name = "__wpos";
      m.lt = LType::BIGINT;
      m.pt = PType::I64;
      cols.push_back(m);
    }
    static std::atomic<int> wseq{0};
    const std::string name = "__tmp" + std::to_string(++wseq) + "_wincol";
    auto t = std::make_shared<Table>(name, cols, -1, -1, std::vector<int>{}, std::vector<int>{}, 1, false);
    t->db = "__tmp";
    std::vector<std::vector<int64_t>> iv(cols.size());
    std::vector<std::vector<double>> dv(cols.size());
    for (size_t c = 0; c < nin; ++c) {
      if (src->cols[c].lt == LType::VARCHAR) t->dicts[c] = src->dicts[c];
      if (isF((int)c)) dv[c].assign((const double*)raw[c], (const double*)raw[c] + n);
      else {
        iv[c].resize(n);
        for (size_t i = 0; i < n; ++i) iv[c][i] = rawI((int)c, i);
      }
    }
    for (size_t w = 0; w < outMeta.size(); ++w) {
      const size_t c = nin + w;
      if (outDict[w]) t->dicts[c] = outDict[w];
      if (outMeta[w].pt == PType::F64) dv[c] = std::move(outD[w]);
      else iv[c] = std::move(outI[w]);
    }
    if (!lastOrderC.empty()) {
      auto& pos = iv[cols.size() - 1];
      pos.resize(n);
      for (size_t j = 0; j < n; ++j) pos[lastOrderC[j]] = (int64_t)j;
      outer.orderBy.push_back({mkCol("__win", "__wpos"), false});
    }
    if (n) {
      t->appendColumns(iv, dv, (int64_t)n);
      t->flush(true);
    }
    {
      std::lock_guard<std::mutex> g(cat_.mu);
      cat_.tables["__tmp." + lower(name)] = t;
      q.temps.push_back("__tmp." + lower(name));
    }
    if (envSetting("TRACE")) fprintf(stderr, "[trace] windows (columnar): %.1f ms, %zu rows\n", msSince(tW0), n);
    outer.from = {TableRef{"__tmp." + name, "__win", nullptr, 0, nullptr}};
    tInsertCols = outerCols;
    return query(outer, q, explain, typed);
  }
  const auto tW1 = Clock::now();

  const size_t n = r.vals.size(), base = r.names.size();
  std::vector<size_t> lastOrder;  // rows in the last window's (PARTITION BY, ORDER BY) order
  for (size_t w = 0; w < wins.size(); ++w) {
    const Win& wn = wins[w];
    const std::string f = wn.node->op;
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    // Sort keys as doubles (strings: rank among the distinct values; NULL lowest), so the sort and
    // the partition / peer scans compare numbers instead of Values.
    std::map<int, std::vector<double>> keys;
    auto keyCol = [&](int col) -> const std::vector<double>& {
      auto it = keys.find(col);
      if (it != keys.end()) return it->second;
      std::vector<double> k(n);
      bool str = false;
      for (size_t i = 0; i < n && !str; ++i) str = r.vals[i][col].kind == Value::STR;
      if (str) {  // rank the distinct strings, then look each row up
        std::unordered_map<std::string_view, size_t> id;
        std::vector<size_t> first;
        std::vector<uint32_t> rowId(n, 0);
        for (size_t i = 0; i < n; ++i) {
          const Value& v = r.vals[i][col];
          if (v.isNull()) continue;
          auto [it, fresh] = id.try_emplace(v.s, first.size());
          if (fresh) first.push_back(i);
          rowId[i] = (uint32_t)it->second;
        }
        std::vector<size_t> o(first.size());
        for (size_t j = 0; j < o.size(); ++j) o[j] = j;
        std::sort(o.begin(), o.end(), [&](size_t a, size_t b) { return cmpVal(r.vals[first[a]][col], r.vals[first[b]][col]) < 0; });
        std::vector<double> rank(o.size());
        double rk = 0;
        for (size_t j = 0; j < o.size(); ++j) {
          if (j && cmpVal(r.vals[first[o[j - 1]]][col], r.vals[first[o[j]]][col]) != 0) ++rk;
          rank[o[j]] = rk;
        }
        for (size_t i = 0; i < n; ++i)
          if (!r.vals[i][col].isNull()) k[i] = rank[rowId[i]];
      }
      parallelFor(n, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i)
          if (r.vals[i][col].isNull()) k[i] = -HUGE_VAL;
          else if (!str) k[i] = numOf(r.vals[i][col]);
      });
      return keys[col] = std::move(k);
    };
    std::vector<const std::vector<double>*> pk;
    std::vector<std::pair<const std::vector<double>*, bool>> ok;
    for (int p : wn.part) pk.push_back(&keyCol(p));
    for (auto& [o, d] : wn.ord) ok.push_back({&keyCol(o), d});
    auto cmpKeys = [&](size_t a, size_t b, bool withOrder) {
      for (auto* k : pk) {
        double x = (*k)[a], y = (*k)[b];
        if (x != y) return x < y ? -1 : 1;
      }
      if (withOrder)
        for (auto& [k, d] : ok) {
          double x = (*k)[a], y = (*k)[b];
          if (x != y) return (x < y) != d ? -1 : 1;
        }
      return 0;
    };
    const auto tK = Clock::now();
    {  // stable sort on all cores: sorted chunks, then pairwise merges
      auto less = [&](size_t a, size_t b) { return cmpKeys(a, b, true) < 0; };
      const size_t nth = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), n / 32768));
      std::vector<size_t> bounds;
      for (size_t t = 0; t <= nth; ++t) bounds.push_back(n * t / nth);
      parallelFor(nth, [&](size_t b, size_t e) {
        for (size_t t = b; t < e; ++t) std::stable_sort(idx.begin() + bounds[t], idx.begin() + bounds[t + 1], less);
      }, 1);
      for (size_t step = 1; step < nth; step *= 2) {
        std::vector<size_t> starts;
        for (size_t t = 0; t + step < nth; t += 2 * step) starts.push_back(t);
        parallelFor(starts.size(), [&](size_t b, size_t e) {
          for (size_t k = b; k < e; ++k) {
            size_t t = starts[k];
            std::inplace_merge(idx.begin() + bounds[t], idx.begin() + bounds[t + step], idx.begin() + bounds[std::min(nth, t + 2 * step)], less);
          }
        }, 1);
      }
    }
    if (envSetting("TRACE")) fprintf(stderr, "[trace] window %zu: keys+sort %.1f ms (sort %.1f)\n", w, msSince(tW1), msSince(tK));
    std::vector<Value> out(n);
    LType ty = LType::BIGINT;
    const LType argTy = wn.args.empty() ? LType::BIGINT : r.types[wn.args[0]];
    if (f == "PERCENT_RANK" || f == "CUME_DIST" || f == "AVG") ty = LType::DOUBLE;
    else if (f == "SUM") ty = argTy == LType::DOUBLE ? LType::DOUBLE : LType::BIGINT;
    else if (f == "MIN" || f == "MAX" || f == "LAG" || f == "LEAD" || f == "FIRST_VALUE" || f == "LAST_VALUE" || f == "NTH_VALUE") ty = argTy;
    auto argv = [&](size_t row, size_t k) -> Value { return k < wn.args.size() ? r.vals[row][wn.args[k]] : Value::Null(); };
    auto constArg = [&](size_t k, double def) {  // LAG/LEAD offset, NTILE buckets, NTH_VALUE n: constants
      if (k >= wn.node->args.size()) return def;
      const ExprP& a = wn.node->args[k];
      return a->kind == Expr::LIT ? numOf(a->lit) : def;
    };
    static const std::set<std::string> kWinFns = {"ROW_NUMBER", "RANK", "DENSE_RANK", "PERCENT_RANK", "CUME_DIST", "NTILE", "LAG", "LEAD",
                                                  "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "SUM", "AVG", "COUNT", "MIN", "MAX"};
    if (!kWinFns.count(f)) throw SqlError("window function " + f + " is not supported", 1235);
    // Partitions are independent: computed on all cores.
    std::vector<std::pair<size_t, size_t>> parts;
    for (size_t a = 0; a < n;) {
      size_t b = a + 1;
      while (b < n && cmpKeys(idx[a], idx[b], false) == 0) ++b;
      parts.push_back({a, b});
      a = b;
    }
    parallelFor(parts.size(), [&](size_t pb, size_t pe) {
    for (size_t pi = pb; pi < pe; ++pi) {
      const size_t a = parts[pi].first, b = parts[pi].second;
      const size_t cnt = b - a;
      // peer groups (equal ORDER BY keys)
      std::vector<size_t> peerStart(cnt), peerEnd(cnt);
      for (size_t i = 0; i < cnt;) {
        size_t j = i + 1;
        while (j < cnt && cmpKeys(idx[a + i], idx[a + j], true) == 0) ++j;
        for (size_t k = i; k < j; ++k) peerStart[k] = i, peerEnd[k] = j;
        i = j;
      }
      int64_t dense = 0;
      for (size_t i = 0; i < cnt; ++i) {
        const size_t row = idx[a + i];
        if (i == 0 || peerStart[i] == i) ++dense;
        // frame [lo, hi) within the partition
        size_t lo, hi;
        if (wn.node->rowsFrame) {
          int64_t L = wn.node->frameLo == INT64_MIN ? 0 : std::max<int64_t>(0, (int64_t)i + wn.node->frameLo);
          int64_t H = wn.node->frameHi == INT64_MAX ? (int64_t)cnt : std::min<int64_t>((int64_t)cnt, (int64_t)i + wn.node->frameHi + 1);
          lo = (size_t)L;
          hi = (size_t)std::max<int64_t>(L, H);
        } else if (wn.ord.empty()) {
          lo = 0, hi = cnt;
        } else {
          lo = 0, hi = wn.node->frameHi == INT64_MAX ? cnt : peerEnd[i];
        }
        Value v;
        if (f == "ROW_NUMBER") v = Value::I((int64_t)i + 1);
        else if (f == "RANK") v = Value::I((int64_t)peerStart[i] + 1);
        else if (f == "DENSE_RANK") v = Value::I(dense);
        else if (f == "PERCENT_RANK") v = Value::D(cnt > 1 ? (double)peerStart[i] / (double)(cnt - 1) : 0.0);
        else if (f == "CUME_DIST") v = Value::D((double)peerEnd[i] / (double)cnt);
        else if (f == "NTILE") {
          int64_t k = std::max<int64_t>(1, (int64_t)constArg(0, 1)), sz = (int64_t)cnt / k, rem = (int64_t)cnt % k;
          int64_t pos = (int64_t)i, bucket = pos < rem * (sz + 1) ? pos / (sz + 1) : rem + (pos - rem * (sz + 1)) / std::max<int64_t>(sz, 1);
          v = Value::I(bucket + 1);
        } else if (f == "LAG" || f == "LEAD") {
          int64_t off = (int64_t)constArg(1, 1), j = f == "LAG" ? (int64_t)i - off : (int64_t)i + off;
          v = j >= 0 && j < (int64_t)cnt ? argv(idx[a + j], 0) : (wn.args.size() > 2 ? argv(row, 2) : Value::Null());
        } else if (f == "FIRST_VALUE") v = hi > lo ? argv(idx[a + lo], 0) : Value::Null();
        else if (f == "LAST_VALUE") v = hi > lo ? argv(idx[a + hi - 1], 0) : Value::Null();
        else if (f == "NTH_VALUE") {
          size_t k = (size_t)std::max<double>(1, constArg(1, 1));
          v = lo + k - 1 < hi ? argv(idx[a + lo + k - 1], 0) : Value::Null();
        } else if (f == "SUM" || f == "AVG" || f == "COUNT" || f == "MIN" || f == "MAX") {
          double acc = 0;
          int64_t c = 0;
          Value best = Value::Null();  // (a default Value is the number 0, not NULL)
          bool star = wn.args.empty();
          for (size_t j = lo; j < hi; ++j) {
            if (star) { ++c; continue; }
            Value x = argv(idx[a + j], 0);
            if (x.isNull()) continue;
            ++c;
            acc += numOf(x);
            if (best.isNull() || (f == "MIN" ? cmpVal(x, best) < 0 : cmpVal(x, best) > 0)) best = x;
          }
          if (f == "COUNT") v = Value::I(c);
          else if (c == 0) v = Value::Null();
          else if (f == "SUM") v = ty == LType::DOUBLE ? Value::D(acc) : Value::I((int64_t)std::llround(acc));
          else if (f == "AVG") v = Value::D(acc / c);
          else v = best;
        } else {
          throw SqlError("window function " + f + " is not supported", 1235);
        }
        out[row] = v;
      }
    }
    }, std::max<size_t>(1, parts.size() / (4 * std::thread::hardware_concurrency())));
    if (w + 1 == wins.size() && s.orderBy.empty() && !s.distinct && (!wn.part.empty() || !wn.ord.empty())) lastOrder = idx;
    r.names.push_back("__w" + std::to_string(w));
    r.types.push_back(ty);
    parallelFor(n, [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) {
        if (w == 0) r.vals[i].reserve(base + wins.size());
        r.vals[i].push_back(std::move(out[i]));
      }
    });
  }
  (void)base;
  if (!lastOrder.empty()) {  // no ORDER BY: rows come out in window order, as in MySQL (ORMs rely on it)
    std::vector<int64_t> pos(n);
    for (size_t j = 0; j < n; ++j) pos[lastOrder[j]] = (int64_t)j;
    r.names.push_back("__wpos");
    r.types.push_back(LType::BIGINT);
    for (size_t i = 0; i < n; ++i) r.vals[i].push_back(Value::I(pos[i]));
    outer.orderBy.push_back({mkCol("__win", "__wpos"), false});
  }
  const auto tW2 = Clock::now();
  std::string tmp = makeTemp(r, q, "win");
  if (envSetting("TRACE"))
    fprintf(stderr, "[trace] windows: inner %.1f ms, compute %.1f ms, temp %.1f ms (%zu rows)\n", msSince(tW0) - msSince(tW1),
            msSince(tW1) - msSince(tW2), msSince(tW2), n);
  outer.from = {TableRef{tmp, "__win", nullptr, 0, nullptr}};
  return query(outer, q, explain, typed);
}

void Engine::stopPipeline(Pipeline& p) {
  p.running = false;
  if (p.th.joinable()) p.th.join();
  std::lock_guard<std::mutex> g(p.mu);
  p.state = "Stopped";
}

void Engine::startPipeline(const std::shared_ptr<Pipeline>& p) {
  if (p->running) return;
  {
    std::lock_guard<std::mutex> g(p->mu);
    for (auto& f : replayedLoads_) p->done.insert(f);
    p->state = "Running";
  }
  p->running = true;
  if (p->glob.rfind("MYSQL\x1e", 0) == 0) {  // MySQL CDC: snapshot, then follow the binlog
    p->th = std::thread([this, p] {
      currentDb() = p->db;
      const std::string url = p->glob.substr(6, p->glob.find('\x1e', 6) - 6);  // MYSQL \x1e url \x1e ..
      try {
        mysqlCdcRun(*this, url, p->table, p->name, p->running, true, [&](const CdcStatus& st) {
          std::lock_guard<std::mutex> lk(p->mu);
          p->state = st.state;
          p->lastError = st.error;
          p->rows = st.rows;
          p->files = st.txns;
        });
      } catch (std::exception& e) {
        std::lock_guard<std::mutex> lk(p->mu);
        p->state = "Error";
        p->lastError = e.what();
      }
    });
    return;
  }
  if (p->glob.find('\x1e') != std::string::npos) {
    p->th = std::thread([this, p] {
      currentDb() = p->db;
      runRemotePipeline(p);
    });
    return;
  }
  p->th = std::thread([this, p] {
    currentDb() = p->db;
    while (p->running) {
      glob_t g{};
      std::vector<std::string> files;
      if (::glob(p->glob.c_str(), 0, nullptr, &g) == 0)
        for (size_t i = 0; i < g.gl_pathc; ++i) files.push_back(g.gl_pathv[i]);
      globfree(&g);
      std::sort(files.begin(), files.end());
      for (auto& f : files) {
        if (!p->running) break;
        {
          std::lock_guard<std::mutex> lk(p->mu);
          if (p->done.count(f)) continue;
        }
        std::string sql = "LOAD DATA INFILE '" + f + "' INTO TABLE " + p->table + " FIELDS TERMINATED BY '" +
                          (p->delim == "\t" ? std::string("\\t") : p->delim) + "' IGNORE " + std::to_string(p->skip) + " LINES" +
                          (p->cols.empty() ? "" : " (" + p->cols + ")");
        try {
          ResultSet r = execute(sql);
          std::lock_guard<std::mutex> lk(p->mu);
          p->done.insert(f);
          ++p->files;
          p->rows += r.affected;
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lk(p->mu);
          p->done.insert(f);  // don't retry a bad file forever
          p->lastError = f + ": " + e.what();
          p->state = "Running (with errors)";
        }
      }
      for (int i = 0; i < 10 && p->running; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });
}

ResultSet Engine::doPipeline(Stmt& s) {
  const std::string key = lower(s.table);
  if (s.setName == "CREATE") {
    const Stmt& l = *s.inner;
    if (l.setValue != "FS" && l.setValue != "S3" && l.setValue != "KAFKA" && l.setValue != "LINK" && l.setValue != "MYSQL")
      throw SqlError("pipelines from " + l.setValue + " are not supported (use FS, S3, KAFKA or MYSQL)", 1235);
    if (l.setValue == "MYSQL" && l.format != "DATABASE") throw SqlError("LOAD DATA MYSQL loads INTO DATABASE <name>", 1064);
    if (l.setValue != "MYSQL") cat_.get(l.table);  // the target table must exist
    auto p = std::make_shared<Pipeline>();
    p->name = s.table;
    p->db = currentDb();
    p->table = l.setValue == "MYSQL" ? lower(l.table) : l.table.find('.') == std::string::npos ? currentDb() + "." + l.table : l.table;
    p->glob = l.path;
    if (l.setValue != "FS" || l.format == "JSON") {  // remote source: packed spec (see unpackSource)
      std::string paths;
      for (auto& jp : l.jsonPaths) paths += (paths.empty() ? "" : "\x1f") + jp;
      p->glob = l.setValue + "\x1e" + l.path + "\x1e" + l.config + "\x1e" + l.credentials + "\x1e" + l.format + "\x1e" + paths;
    }
    p->delim = std::string(1, l.delim);
    p->skip = l.skipLines;
    for (auto& c : l.insertCols) p->cols += (p->cols.empty() ? "" : ", ") + c;
    std::lock_guard<std::mutex> g(pipesMu_);
    if (pipes_.count(key)) {
      if (s.ifNotExists) return okResult(0);
      throw SqlError("Pipeline '" + s.table + "' already exists", 1050);
    }
    pipes_[key] = p;
    return okResult(0);
  }
  std::shared_ptr<Pipeline> p;
  {
    std::lock_guard<std::mutex> g(pipesMu_);
    auto it = pipes_.find(key);
    if (it == pipes_.end()) {
      if (s.ifExists) return okResult(0);
      throw SqlError("Pipeline '" + s.table + "' doesn't exist", 1146);
    }
    p = it->second;
  }
  if (s.setName == "START") {
    p->wantRun = true;
    if (!replaying_ && !isReplica_) startPipeline(p);  // replicas get the pipeline's loads from the WAL
  } else if (s.setName == "STOP") {
    p->wantRun = false;
    stopPipeline(*p);
  } else if (s.setName == "DROP") {
    stopPipeline(*p);
    std::lock_guard<std::mutex> g(pipesMu_);
    pipes_.erase(key);
  }
  return okResult(0);
}

namespace {
struct SourceSpec {
  std::string src, path, config, creds, format;
  std::vector<std::string> jsonPaths;
};
SourceSpec unpackSource(const std::string& g) {
  std::vector<std::string> f;
  for (size_t a = 0;;) {
    size_t b = g.find('\x1e', a);
    f.push_back(g.substr(a, b == std::string::npos ? std::string::npos : b - a));
    if (b == std::string::npos) break;
    a = b + 1;
  }
  f.resize(6);
  SourceSpec s{f[0], f[1], f[2], f[3], f[4], {}};
  for (size_t a = 0; !f[5].empty();) {
    size_t b = f[5].find('\x1f', a);
    s.jsonPaths.push_back(f[5].substr(a, b == std::string::npos ? std::string::npos : b - a));
    if (b == std::string::npos) break;
    a = b + 1;
  }
  return s;
}
std::string sanitizeName(const std::string& s) {
  std::string o;
  for (char c : s) o += isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_' ? c : '_';
  return o.size() > 180 ? o.substr(o.size() - 180) : o;
}
std::string baseName(const std::string& p) { return p.substr(p.rfind('/') + 1); }
}  // namespace

int64_t Engine::ingest(const std::string& table, char delim, int64_t skip, const std::vector<std::string>& cols,
                       const std::vector<std::string>& jsonPaths, bool json, const std::string& data, const std::string& stagedName) {
  std::string body = data;
  std::vector<std::string> loadCols = cols;
  bool escaped = false;  // the staged text uses backslash escapes (Parquet values can hold tabs / newlines)
  const bool parquet = data.size() >= 12 && data.compare(0, 4, "PAR1") == 0 && data.compare(data.size() - 4, 4, "PAR1") == 0;
  const bool avro = data.size() >= 4 && data.compare(0, 4, std::string("Obj\x01", 4)) == 0;
  const bool orc = data.size() >= 16 && data.compare(0, 3, "ORC") == 0;
  if (parquet) {  // staged as it is: LOAD DATA reads Parquet directly (doLoad -> loadParquet), by column name
    delim = '\t';
    skip = 0;
  } else if (avro || orc) {  // row / columnar binary formats: as text, by column name
    std::vector<std::string> names = cols;
    if (names.empty()) {
      auto t = cat_.get(table);
      for (auto& c : t->cols) names.push_back(c.name);
    }
    body = avro ? avroToTsv(data, names) : orcToTsv(data, names);
    delim = '\t';
    skip = 0;
    escaped = true;
  } else if (json) {
    std::vector<std::string> paths = jsonPaths;
    if (cols.empty()) {
      auto t = cat_.get(table);
      for (auto& c : t->cols) paths.push_back(c.name);
    }
    body = jsonLinesToTsv(data, paths);
    delim = '\t';
    skip = 0;
  }
  const std::string dir = dir_.empty() ? std::string("/tmp") : dir_ + "/pipeline-loads";
  if (!dir_.empty()) mkdir(dir.c_str(), 0755);
  const std::string path = dir + "/" + (dir_.empty() ? "warpstore-" : "") + stagedName + ".tsv";  // (never re-routed by its name)
  {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw SqlError("cannot stage " + path, 1105);
    fwrite(body.data(), 1, body.size(), f);
    fclose(f);
  }
  std::string colList;
  for (auto& c : loadCols) colList += (colList.empty() ? "" : ", ") + c;
  std::string sql = "LOAD DATA INFILE '" + path + "' INTO TABLE " + table + " FIELDS TERMINATED BY '" +
                    (delim == '\t' ? std::string("\\t") : std::string(1, delim)) + "'" + (escaped ? " ESCAPED BY '\\\\'" : "") +
                    " IGNORE " + std::to_string(skip) + " LINES" +
                    (colList.empty() ? "" : " (" + colList + ")");
  ResultSet r;
  try {
    r = execute(sql);
  } catch (...) {
    if (dir_.empty()) unlink(path.c_str());
    throw;
  }
  if (dir_.empty()) unlink(path.c_str());  // without a data dir nothing replays it
  return r.affected;
}

ResultSet Engine::loadRemote(Stmt& s) {
  auto t0 = Clock::now();
  const bool json = s.format == "JSON";
  const std::string table = s.table.find('.') == std::string::npos ? currentDb() + "." + s.table : s.table;
  if (s.setValue != "MYSQL") cat_.get(table);
  int64_t total = 0, objects = 0;
  const std::string tag = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
  if (s.setValue == "MYSQL") {  // one-off copy of a MySQL database (a pipeline keeps following it)
    std::atomic<bool> run{true};
    CdcStatus last;
    mysqlCdcRun(*this, s.path, s.table, "load-" + std::to_string(nowUs()), run, false, [&](const CdcStatus& st) { last = st; });
    return okResult(last.rows, "copied " + std::to_string(last.rows) + " rows from MySQL (binlog position " + last.position + ")");
  }
  if (s.setValue == "S3") {
    const std::string bucket = s3Bucket(s.path);
    for (auto& o : s3List(s.path, s.config, s.credentials)) {
      std::string data = decompressByName(o.key, s3Get(bucket, o.key, s.config, s.credentials));
      total += ingest(table, s.delim, s.skipLines, s.insertCols, s.jsonPaths, json, data, "s3-" + tag + "-" + sanitizeName(o.key));
      ++objects;
    }
  } else if (s.setValue == "KAFKA") {
    std::map<int, int64_t> next;
    for (int round = 0; round < 100000; ++round) {
      auto msgs = kafkaPoll(s.path, next, 8 << 20);
      if (msgs.empty()) break;
      std::string data;
      for (auto& m : msgs) {
        data += m.value;
        if (data.empty() || data.back() != '\n') data += '\n';
      }
      total += ingest(table, s.delim, 0, s.insertCols, s.jsonPaths, json, data, "kafka-" + tag + "-" + std::to_string(round));
      objects += (int64_t)msgs.size();
    }
  } else if (s.path.rfind("http://", 0) == 0 || s.path.rfind("https://", 0) == 0) {
    std::string data = decompressByName(s.path, httpGet(s.path));
    total = ingest(table, s.delim, s.skipLines, s.insertCols, s.jsonPaths, json, data, "http-" + tag);
    objects = 1;
  } else {  // FS + FORMAT JSON
    glob_t g{};
    std::vector<std::string> files;
    if (::glob(s.path.c_str(), 0, nullptr, &g) == 0)
      for (size_t i = 0; i < g.gl_pathc; ++i) files.push_back(g.gl_pathv[i]);
    globfree(&g);
    if (files.empty()) throw SqlError("File '" + s.path + "' not found", 29);
    for (auto& f : files) {
      FILE* fp = fopen(f.c_str(), "rb");
      if (!fp) throw SqlError("cannot read " + f, 29);
      std::string data;
      char buf[1 << 16];
      for (size_t n; (n = fread(buf, 1, sizeof buf, fp)) > 0;) data.append(buf, n);
      fclose(fp);
      total += ingest(table, s.delim, s.skipLines, s.insertCols, s.jsonPaths, json, decompressByName(f, data),
                      "fs-" + tag + "-" + sanitizeName(baseName(f)));
      ++objects;
    }
  }
  char info[160];
  snprintf(info, sizeof info, "Records: %lld  Sources: %lld  Loaded in %.1f s", (long long)total, (long long)objects, msSince(t0) / 1000);
  return okResult(total, info);
}

void Engine::runRemotePipeline(const std::shared_ptr<Pipeline>& p) {
  const SourceSpec sp = unpackSource(p->glob);
  const bool json = sp.format == "JSON";
  std::vector<std::string> cols;
  for (size_t a = 0; !p->cols.empty();) {
    size_t b = p->cols.find(", ", a);
    cols.push_back(p->cols.substr(a, b == std::string::npos ? std::string::npos : b - a));
    if (b == std::string::npos) break;
    a = b + 2;
  }
  const char delim = p->delim.empty() ? '\t' : p->delim[0];
  // Kafka offsets come back from the staged file names already loaded: kafka-<pipe>-<part>-<from>-<next>.
  std::map<int, int64_t> next;
  const std::string kpre = "kafka-" + sanitizeName(lower(p->name)) + "-";
  {
    std::lock_guard<std::mutex> lk(p->mu);
    for (auto& d : p->done) {
      std::string b = baseName(d);
      if (b.rfind("warpstore-", 0) == 0) b = b.substr(6);
      if (b.rfind(kpre, 0) != 0) continue;
      int part;
      long long from, nx;
      if (sscanf(b.c_str() + kpre.size(), "%d-%lld-%lld", &part, &from, &nx) == 3) next[part] = std::max<int64_t>(next[part], nx);
    }
  }
  auto stagedPath = [&](const std::string& name) {  // the file ingest() stages (and the WAL names)
    return (dir_.empty() ? std::string("/tmp/warpstore-") : dir_ + "/pipeline-loads/") + name + ".tsv";
  };
  auto fail = [&](const std::string& what) {
    std::lock_guard<std::mutex> lk(p->mu);
    p->lastError = what;
    p->state = "Running (with errors)";
  };
  while (p->running) {
    try {
      if (sp.src == "S3" || sp.src == "FS") {
        std::vector<std::string> keys;
        if (sp.src == "S3") {
          for (auto& o : s3List(sp.path, sp.config, sp.creds)) keys.push_back(o.key);
        } else {
          glob_t g{};
          if (::glob(sp.path.c_str(), 0, nullptr, &g) == 0)
            for (size_t i = 0; i < g.gl_pathc; ++i) keys.push_back(g.gl_pathv[i]);
          globfree(&g);
        }
        std::sort(keys.begin(), keys.end());
        for (auto& k : keys) {
          if (!p->running) break;
          const std::string name = lower(sp.src) + "-" + sanitizeName(lower(p->name)) + "-" + sanitizeName(k);
          {
            std::lock_guard<std::mutex> lk(p->mu);
            const std::string sp2 = stagedPath(name);
            if (p->done.count(sp2) || p->done.count(sp2.substr(0, sp2.size() - 4))) continue;  // (older snapshots: no .tsv)
          }
          try {
            std::string data;
            if (sp.src == "S3") {
              data = s3Get(s3Bucket(sp.path), k, sp.config, sp.creds);
            } else {
              FILE* fp = fopen(k.c_str(), "rb");
              if (!fp) continue;
              char buf[1 << 16];
              for (size_t n; (n = fread(buf, 1, sizeof buf, fp)) > 0;) data.append(buf, n);
              fclose(fp);
            }
            int64_t n = ingest(p->table, delim, p->skip, cols, sp.jsonPaths, json, decompressByName(k, data), name);
            std::lock_guard<std::mutex> lk(p->mu);
            p->done.insert(stagedPath(name));
            ++p->files;
            p->rows += n;
          } catch (std::exception& e) {
            std::lock_guard<std::mutex> lk(p->mu);
            p->done.insert(stagedPath(name));  // don't retry a bad object forever
            p->lastError = k + ": " + e.what();
            p->state = "Running (with errors)";
          }
        }
      } else if (sp.src == "KAFKA") {
        auto before = next;
        auto msgs = kafkaPoll(sp.path, next, 8 << 20);
        std::map<int, std::string> byPart;
        std::map<int, int64_t> cnt;
        for (auto& m : msgs) {
          std::string& d = byPart[m.partition];
          d += m.value;
          if (d.empty() || d.back() != '\n') d += '\n';
          ++cnt[m.partition];
        }
        for (auto& [part, data] : byPart) {
          int64_t from = before.count(part) ? before[part] : 0;
          const std::string name = kpre + std::to_string(part) + "-" + std::to_string(from) + "-" + std::to_string(next[part]);
          int64_t n = ingest(p->table, delim, 0, cols, sp.jsonPaths, json, data, name);
          std::lock_guard<std::mutex> lk(p->mu);
          for (auto it = p->done.begin(); it != p->done.end();)  // keep only the latest batch per partition
            it = baseName(*it).find(kpre + std::to_string(part) + "-") != std::string::npos ? p->done.erase(it) : std::next(it);
          p->done.insert(stagedPath(name));
          p->files += cnt[part];
          p->rows += n;
        }
        if (!msgs.empty()) continue;  // more may be waiting
      }
      {
        std::lock_guard<std::mutex> lk(p->mu);
        if (p->state == "Running (with errors)" && p->lastError.empty()) p->state = "Running";
      }
    } catch (std::exception& e) {
      fail(e.what());
    }
    for (int i = 0; i < 10 && p->running; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

ResultSet Engine::showPipelines() {
  ResultSet r;
  r.isResult = true;
  r.names = {"Pipelines_in_" + currentDb(), "State", "Table", "Source", "Files_Loaded", "Rows_Loaded", "Last_Error"};
  r.types = {LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::VARCHAR, LType::BIGINT, LType::BIGINT, LType::VARCHAR};
  std::lock_guard<std::mutex> g(pipesMu_);
  for (auto& [k, p] : pipes_) {
    std::lock_guard<std::mutex> lk(p->mu);
    std::string src = p->glob.find('\x1e') == std::string::npos ? "FS " + p->glob
                      : p->glob.substr(0, p->glob.find('\x1e')) + " " + p->glob.substr(p->glob.find('\x1e') + 1, p->glob.find('\x1e', p->glob.find('\x1e') + 1) - p->glob.find('\x1e') - 1);
    if (size_t sch = src.find("://"); sch != std::string::npos)  // never show a password in the source URL
      if (size_t at = src.find('@', sch); at != std::string::npos)
        if (size_t colon = src.find(':', sch + 3); colon != std::string::npos && colon < at) src.replace(colon + 1, at - colon - 1, "***");
    r.rows.push_back({p->name, p->state, p->table, src, std::to_string(p->files), std::to_string(p->rows),
                      p->lastError.empty() ? kNullCell : p->lastError});
  }
  return r;
}

// Host aggregation: the GPU produces the filtered / joined rows (group keys and aggregate arguments),
// the host groups them and computes every aggregate, then HAVING / ORDER BY / LIMIT run over a temp
// table of the groups. Used for aggregates without a GPU implementation and as a fallback.
// GROUP BY with more keys than the GPU path takes: plain INT / BIGINT columns of one table whose value
// ranges are known are packed into one BIGINT key (each value's offset from its minimum, NULL as one
// more value, times the product of the ranges before it); select list, HAVING and ORDER BY read them
// back from it. False if that doesn't bring the keys down to kMaxKeys.
// GROUP BY keys determined by other keys (functional dependencies): a table's columns by its primary
// key, and a column by one it equals in WHERE / inner-join conditions. They are dropped from GROUP BY
// and read as MIN(col) (one value per group): GROUP BY c_custkey, c_name, n_name .. groups on the key
// alone (fewer keys: cheaper hashing, a dense key range, no compile-only plans).
void Engine::dropDependentKeys(Stmt& s) {
  if (s.groupBy.size() < 2 || !s.groupingSets.empty() || s.from.empty()) return;
  for (auto& it : s.items)
    if (it.e->kind == Expr::STAR || it.e->window) return;
  struct T {
    std::shared_ptr<Table> t;
    std::string alias;
  };
  std::vector<T> tabs;
  for (auto& f : s.from) {
    T x;
    if (!f.sub && !f.tvf) x.t = cat_.find(f.name.find('.') == std::string::npos ? currentDb() + "." + f.name : f.name);
    x.alias = f.alias.empty() ? f.name : f.alias;
    tabs.push_back(x);
  }
  using TC = std::pair<int, int>;
  auto colOf = [&](const ExprP& e) -> TC {  // a plain column of a FROM table, or (-1, -1)
    if (!e || e->kind != Expr::COL) return {-1, -1};
    TC r{-1, -1};
    for (size_t i = 0; i < tabs.size(); ++i) {
      if (!e->table.empty() && strcasecmp(e->table.c_str(), tabs[i].alias.c_str()) != 0) continue;
      if (!tabs[i].t) {
        if (e->table.empty()) return {-1, -1};  // (a derived table might have it: ambiguous)
        continue;
      }
      const int c = tabs[i].t->colIndex(e->name);
      if (c < 0) continue;
      if (r.first >= 0) return {-1, -1};  // ambiguous
      r = {(int)i, c};
    }
    return r;
  };
  auto resolve = [&](const ExprP& g) -> ExprP {  // GROUP BY 2 / GROUP BY alias
    if (g->kind == Expr::LIT && g->lit.kind == Value::INT && g->lit.i >= 1 && g->lit.i <= (int64_t)s.items.size()) return s.items[g->lit.i - 1].e;
    if (g->kind == Expr::COL && g->table.empty())
      for (auto& it : s.items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), g->name.c_str()) == 0) return it.e;
    return g;
  };
  std::vector<TC> key(s.groupBy.size());
  bool anyPk = false;
  for (size_t i = 0; i < s.groupBy.size(); ++i) key[i] = colOf(resolve(s.groupBy[i]));
  for (auto& x : tabs) anyPk = anyPk || (x.t && !x.t->pkCols.empty());
  if (!anyPk) return;
  // Equalities between columns: WHERE and inner-join conditions (top-level AND).
  std::vector<std::pair<TC, TC>> eqs;
  std::function<void(const ExprP&)> conj = [&](const ExprP& x) {
    if (!x) return;
    if (x->kind == Expr::BIN && x->op == "AND") { conj(x->args[0]); conj(x->args[1]); return; }
    if (x->kind == Expr::BIN && x->op == "=") {
      const TC a = colOf(x->args[0]), b = colOf(x->args[1]);
      if (a.first >= 0 && b.first >= 0) eqs.push_back({a, b});
    }
  };
  conj(s.where);
  for (auto& j : s.joinConds) conj(j);
  for (size_t i = 0; i < s.from.size(); ++i)
    if (s.from[i].join == 0) conj(s.from[i].on);
  auto closure = [&](const std::vector<bool>& use) {
    std::set<TC> d;
    std::vector<bool> full(tabs.size());
    for (size_t i = 0; i < key.size(); ++i)
      if (use[i] && key[i].first >= 0) d.insert(key[i]);
    for (bool changed = true; changed;) {
      changed = false;
      for (auto& [a, b] : eqs) {
        if (d.count(a) && d.insert(b).second) changed = true;
        if (d.count(b) && d.insert(a).second) changed = true;
      }
      for (size_t t = 0; t < tabs.size(); ++t) {
        if (full[t] || !tabs[t].t || tabs[t].t->pkCols.empty()) continue;
        bool all = true;
        for (int c : tabs[t].t->pkCols) all = all && d.count({(int)t, c});
        if (!all) continue;
        full[t] = true;
        changed = true;
        for (int c = 0; c < (int)tabs[t].t->cols.size(); ++c) d.insert({(int)t, c});
      }
    }
    return d;
  };
  std::vector<bool> use(key.size(), true);
  int dropped = 0;
  for (size_t i = key.size(); i-- > 0;) {
    if (key[i].first < 0) continue;
    use[i] = false;
    if (closure(use).count(key[i])) ++dropped;
    else use[i] = true;
  }
  if (!dropped) return;
  int aggs = dropped;
  std::function<void(const ExprP&)> countAggs = [&](const ExprP& x) {
    if (!x) return;
    if (x->kind == Expr::FUNC && (isAggName(x->op) || isHostAggName(x->op))) { aggs += 1 + (x->op == "AVG"); return; }
    for (auto& a : x->args) countAggs(a);
  };
  for (auto& it : s.items) countAggs(it.e);
  countAggs(s.having);
  if (aggs > kInterpAggs - 2) return;
  std::set<TC> gone;
  for (size_t i = 0; i < key.size(); ++i)
    if (!use[i]) gone.insert(key[i]);
  for (size_t i = 0; i < key.size(); ++i)
    if (use[i]) gone.erase(key[i]);  // (a repeated key: still grouped on)
  bool mixed = false;  // a kept expression key over a dropped column: left alone
  std::function<void(const ExprP&)> refs = [&](const ExprP& x) {
    if (!x || mixed) return;
    if (x->kind == Expr::COL && gone.count(colOf(x))) mixed = true;
    for (auto& a : x->args) refs(a);
  };
  for (size_t i = 0; i < key.size(); ++i)
    if (use[i] && key[i].first < 0) refs(resolve(s.groupBy[i]));
  if (mixed) return;
  for (size_t i = 0; i < key.size(); ++i)
    if (use[i]) s.groupBy[i] = cloneExpr(resolve(s.groupBy[i]));  // (an alias / ordinal: its item is rewritten below)
  std::function<void(ExprP&)> wrap = [&](ExprP& x) {
    if (!x || x->kind == Expr::SUBQUERY || x->sub) return;
    if (x->kind == Expr::FUNC && (isAggName(x->op) || isHostAggName(x->op))) return;
    if (x->kind == Expr::COL && gone.count(colOf(x))) {
      auto m = std::make_shared<Expr>();
      m->kind = Expr::FUNC;
      m->op = "MIN";
      m->args = {x};
      x = m;
      return;
    }
    for (auto& a : x->args) wrap(a);
  };
  for (auto& it : s.items) {
    const std::string before = it.e->str(), name = it.e->kind == Expr::COL ? it.e->name : before;
    wrap(it.e);
    if (it.alias.empty() && it.e->str() != before) it.alias = name;  // (the output column keeps its name)
  }
  wrap(s.having);
  for (auto& o : s.orderBy) {
    if (o.e->kind == Expr::LIT) continue;  // ORDER BY 2
    bool alias = false;
    if (o.e->kind == Expr::COL && o.e->table.empty())
      for (auto& it : s.items) alias = alias || strcasecmp(it.alias.c_str(), o.e->name.c_str()) == 0;
    if (!alias) wrap(o.e);
  }
  std::vector<ExprP> keep;
  for (size_t i = 0; i < key.size(); ++i)
    if (use[i]) keep.push_back(s.groupBy[i]);
  s.groupBy = keep;
}

bool Engine::packGroupKeys(Stmt& s) {
  if (s.from.size() != 1 || s.from[0].sub || !s.joinConds.empty() || s.unionNext) return false;
  auto t = cat_.find(s.from[0].name.find('.') == std::string::npos ? currentDb() + "." + s.from[0].name : s.from[0].name);
  if (!t) return false;
  const std::string alias = s.from[0].alias.empty() ? s.from[0].name : s.from[0].alias;
  auto resolve = [&](const ExprP& g) -> ExprP {  // GROUP BY 2 / GROUP BY alias
    if (g->kind == Expr::LIT && g->lit.kind == Value::INT && g->lit.i >= 1 && g->lit.i <= (int64_t)s.items.size()) return s.items[g->lit.i - 1].e;
    if (g->kind == Expr::COL && g->table.empty())
      for (auto& it : s.items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), g->name.c_str()) == 0) return it.e;
    return g;
  };
  struct Key {
    int col;
    int64_t lo;
    uint64_t width;  // values + 1 (NULL)
  };
  std::vector<Key> pack;
  std::vector<ExprP> rest;
  long double product = 1;
  for (auto& g0 : s.groupBy) {
    const ExprP g = resolve(g0);
    int c = -1;
    if (g->kind == Expr::COL && (g->table.empty() || strcasecmp(g->table.c_str(), alias.c_str()) == 0)) c = t->colIndex(g->name);
    int64_t lo, hi;
    bool dup = false;
    for (auto& k : pack) dup = dup || k.col == c;
    if (c >= 0 && !dup && (t->cols[c].lt == LType::INT || t->cols[c].lt == LType::BIGINT) && colRange(*t, c, lo, hi) && hi >= lo &&
        (uint64_t)hi - (uint64_t)lo < ((uint64_t)1 << 40) && product * ((long double)((uint64_t)hi - (uint64_t)lo) + 2) < 9.0e15L) {
      pack.push_back({c, lo, (uint64_t)hi - (uint64_t)lo + 2});
      product *= (long double)pack.back().width;
    } else {
      rest.push_back(g0);
    }
  }
  if (pack.size() < 2 || rest.size() + 1 > (size_t)kMaxKeys) return false;
  auto q = [&](int c) { return "`" + t->cols[c].name + "`"; };
  std::string p;
  uint64_t mult = 1;
  std::vector<uint64_t> mults;
  for (auto& k : pack) {
    mults.push_back(mult);
    p += (p.empty() ? "" : " + ") + std::string("COALESCE(") + q(k.col) + " - (" + std::to_string(k.lo) + "), " + std::to_string(k.width - 1) + ") * " + std::to_string(mult);
    mult *= k.width;
  }
  p = "(" + p + ")";
  auto parseExpr = [](const std::string& x) { return parseSql("SELECT " + x).items.at(0).e; };
  std::vector<ExprP> decode;
  for (size_t i = 0; i < pack.size(); ++i) {
    const std::string slot = "((" + p + " DIV " + std::to_string(mults[i]) + ") % " + std::to_string(pack[i].width) + ")";
    decode.push_back(parseExpr("IF(" + slot + " = " + std::to_string(pack[i].width - 1) + ", NULL, " + slot + " + (" + std::to_string(pack[i].lo) + "))"));
  }
  std::function<void(ExprP&)> rw = [&](ExprP& x) {
    if (!x || x->kind == Expr::SUBQUERY || x->sub) return;
    if (x->kind == Expr::COL && (x->table.empty() || strcasecmp(x->table.c_str(), alias.c_str()) == 0)) {
      const int c = t->colIndex(x->name);
      for (size_t i = 0; i < pack.size(); ++i)
        if (pack[i].col == c) {
          x = cloneExpr(decode[i]);
          return;
        }
      return;
    }
    if (x->kind == Expr::FUNC && (isAggName(x->op) || isHostAggName(x->op))) return;  // (aggregates read the rows)
    for (auto& a : x->args) rw(a);
  };
  for (auto& it : s.items) {
    if (it.alias.empty() && it.e->kind == Expr::COL) it.alias = it.e->name;  // (keep the column header)
    rw(it.e);
  }
  rw(s.having);
  for (auto& o : s.orderBy) {
    if (o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT) continue;  // ORDER BY 1: the item
    rw(o.e);
  }
  s.groupBy = rest;
  s.groupBy.insert(s.groupBy.begin(), parseExpr(p));
  return true;
}

// GROUP BY with DISTINCT aggregates over different arguments: one GPU aggregation for the plain
// aggregates and one per DISTINCT aggregate (each the usual single-DISTINCT path), merged per group
// key (NULL keys included) into a temp table; the select list, HAVING, ORDER BY and LIMIT then run
// over it.
ResultSet Engine::multiDistinct(Stmt& s, QueryCtx& q, bool typed) {
  auto resolve = [&](const ExprP& g) -> ExprP {
    if (g->kind == Expr::LIT && g->lit.kind == Value::INT && g->lit.i >= 1 && g->lit.i <= (int64_t)s.items.size()) return s.items[g->lit.i - 1].e;
    if (g->kind == Expr::COL && g->table.empty())
      for (auto& it : s.items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), g->name.c_str()) == 0) return it.e;
    return g;
  };
  std::vector<ExprP> keys;
  for (auto& g : s.groupBy) keys.push_back(cloneExpr(resolve(g)));
  std::vector<ExprP> plain, dist;
  std::map<std::string, std::string> colOf;  // aggregate text -> merged column
  std::function<void(const ExprP&)> collect = [&](const ExprP& e) {
    if (!e || e->kind == Expr::SUBQUERY || e->sub) return;
    if (e->kind == Expr::FUNC && e->window) throw SqlError("window", 1235);
    if (e->kind == Expr::FUNC && (isAggName(e->op) || isHostAggName(e->op))) {
      const std::string k = e->str();
      if (!colOf.count(k)) {
        if (e->distinct) colOf[k] = "__d" + std::to_string(dist.size()), dist.push_back(e);
        else colOf[k] = "__a" + std::to_string(plain.size()), plain.push_back(e);
      }
      return;
    }
    for (auto& a : e->args) collect(a);
  };
  for (auto& it : s.items) collect(it.e);
  collect(s.having);
  for (auto& o : s.orderBy) collect(o.e);
  auto part = [&](const std::vector<ExprP>& aggs, const std::string& prefix) {
    Stmt p;
    p.type = Stmt::SELECT;
    p.from = s.from;
    p.joinConds = s.joinConds;
    p.where = s.where ? cloneExpr(s.where) : nullptr;
    for (size_t i = 0; i < keys.size(); ++i) {
      p.items.push_back({cloneExpr(keys[i]), "__g" + std::to_string(i)});
      p.groupBy.push_back(cloneExpr(keys[i]));
    }
    for (size_t j = 0; j < aggs.size(); ++j) p.items.push_back({cloneExpr(aggs[j]), prefix + std::to_string(j)});
    return p;
  };
  auto keyOf = [&](const std::vector<Value>& row) {  // (numbers by value: the parts may type a key differently, 19 / 19.0)
    std::string k;
    char b[40];
    for (size_t i = 0; i < keys.size(); ++i) {
      const Value& v = row[i];
      if (v.isNull()) k += "\x02";
      else if (v.kind == Value::STR) k += v.s;
      else snprintf(b, sizeof b, "%.17g", v.num()), k += b;
      k += '\x1f';
    }
    return k;
  };
  Stmt base = part(plain, "__a");
  ResultSet m = query(base, q, false, true);
  std::unordered_map<std::string, size_t> at;
  for (size_t r = 0; r < m.vals.size(); ++r) at.emplace(keyOf(m.vals[r]), r);
  for (size_t k = 0; k < dist.size(); ++k) {
    Stmt dk = part({dist[k]}, "__v");
    ResultSet rk = query(dk, q, false, true);
    for (auto& row : m.vals) row.push_back(Value::Null());
    for (auto& row : rk.vals) {
      auto it = at.find(keyOf(row));
      if (it != at.end()) m.vals[it->second].back() = row[keys.size()];
    }
    m.names.push_back("__d" + std::to_string(k));
    m.types.push_back(rk.types.at(keys.size()));
  }
  const std::string tmp = makeTemp(m, q, "mdist");
  // the original query over the merged groups
  std::function<ExprP(const ExprP&)> rw = [&](const ExprP& e) -> ExprP {
    if (!e) return e;
    if (e->kind == Expr::SUBQUERY || e->sub) return e;
    const std::string k = e->str();
    for (size_t i = 0; i < keys.size(); ++i)
      if (keys[i]->str() == k) return mkCol("__mg", "__g" + std::to_string(i));
    if (auto it = colOf.find(k); it != colOf.end() && e->kind == Expr::FUNC) return mkCol("__mg", it->second);
    auto y = std::make_shared<Expr>(*e);
    y->bound = false;
    for (auto& a : y->args) a = rw(a);
    return y;
  };
  Stmt outer;
  outer.type = Stmt::SELECT;
  for (auto& it : s.items) outer.items.push_back({rw(it.e), it.alias.empty() ? (it.e->kind == Expr::COL ? it.e->name : it.e->str()) : it.alias});
  outer.from = {TableRef{tmp, "__mg", nullptr, 0, nullptr}};
  outer.where = rw(s.having);
  for (auto& o : s.orderBy) {
    const bool pos = o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT;
    const bool alias = o.e->kind == Expr::COL && o.e->table.empty() &&
                       std::any_of(s.items.begin(), s.items.end(), [&](const SelectItem& it) { return !it.alias.empty() && strcasecmp(it.alias.c_str(), o.e->name.c_str()) == 0; });
    outer.orderBy.push_back({pos || alias ? o.e : rw(o.e), o.desc});
  }
  outer.limit = s.limit;
  outer.offset = s.offset;
  outer.distinct = s.distinct;
  return query(outer, q, false, typed);
}

ResultSet Engine::hostAggregate(Stmt& s, QueryCtx& q, bool explain, bool typed) {
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
  std::vector<ExprP> aggs;
  std::map<std::string, int> aggOf;
  std::function<void(const ExprP&)> collect = [&](const ExprP& e) {
    if (!e) return;
    if (e->kind == Expr::FUNC && (isAggName(e->op) || isHostAggName(e->op)) && !e->window) {
      if (!aggOf.count(e->str() + (e->distinct ? "!" : ""))) {
        aggOf[e->str() + (e->distinct ? "!" : "")] = (int)aggs.size();
        aggs.push_back(e);
      }
      return;
    }
    for (auto& a : e->args) collect(a);
  };
  for (auto& it : s.items) collect(it.e);
  collect(s.having);
  for (auto& o : s.orderBy) collect(o.e);
  // Inner query: group keys and every argument / ORDER BY expression of every aggregate.
  Stmt inner = cloneStmt(s);
  inner.items.clear();
  inner.groupBy.clear();
  inner.having.reset();
  inner.orderBy.clear();
  inner.limit = -1;
  inner.offset = 0;
  inner.distinct = false;
  for (size_t i = 0; i < groups.size(); ++i) inner.items.push_back({cloneExpr(groups[i]), "__g" + std::to_string(i)});
  struct AggCols {
    std::vector<int> args;
    std::vector<std::pair<int, bool>> ord;
  };
  std::vector<AggCols> ac(aggs.size());
  for (size_t j = 0; j < aggs.size(); ++j) {
    for (auto& a : aggs[j]->args) {
      if (a->kind == Expr::STAR) continue;
      ac[j].args.push_back((int)inner.items.size());
      inner.items.push_back({cloneExpr(a), "__x" + std::to_string(inner.items.size())});
    }
    for (auto& [o, d] : aggs[j]->winOrder) {
      ac[j].ord.push_back({(int)inner.items.size(), d});
      inner.items.push_back({cloneExpr(o), "__x" + std::to_string(inner.items.size())});
    }
  }
  if (inner.items.empty()) inner.items.push_back({mkLit(Value::I(1)), "__x0"});
  ResultSet r = query(inner, q, false, true);
  if (r.vals.size() > 50'000'000) throw SqlError("too many rows for host aggregation", 1105);
  // Group.
  std::unordered_map<std::string, size_t> gi;
  std::vector<std::vector<size_t>> members;
  std::vector<size_t> firstRow;
  for (size_t i = 0; i < r.vals.size(); ++i) {
    std::string k;
    for (size_t g = 0; g < groups.size(); ++g) {
      const Value& v = r.vals[i][g];
      k += (v.isNull() ? std::string("\x02") : v.kind == Value::STR && ciColl() ? foldCase(v.s) : v.toString()) + '\x1f';
    }
    auto [it, fresh] = gi.emplace(k, members.size());
    if (fresh) {
      members.emplace_back();
      firstRow.push_back(i);
    }
    members[it->second].push_back(i);
  }
  if (groups.empty() && members.empty()) {  // aggregate over no rows: one group
    members.emplace_back();
    firstRow.push_back(SIZE_MAX);
  }
  auto jsonVal = [](const Value& v) -> std::string {
    if (v.isNull()) return "null";
    if (v.kind == Value::INT || v.kind == Value::DBL) return v.toString();
    std::string o = "\"";
    for (char ch : v.toString()) {
      if (ch == '"' || ch == '\\') o += '\\';
      o += ch;
    }
    return o + "\"";
  };
  ResultSet out;
  out.isResult = true;
  out.typed = true;
  for (size_t g = 0; g < groups.size(); ++g) {
    out.names.push_back("__g" + std::to_string(g));
    out.types.push_back(r.types[g] == LType::DECIMAL ? LType::DOUBLE : r.types[g]);
  }
  for (size_t j = 0; j < aggs.size(); ++j) {
    const std::string f = aggs[j]->op;
    LType at = ac[j].args.empty() ? LType::BIGINT : r.types[ac[j].args[0]];
    if (at == LType::DECIMAL) at = LType::DOUBLE;
    LType t = f == "COUNT" ? LType::BIGINT
            : f == "SUM" ? (at == LType::DOUBLE ? LType::DOUBLE : LType::BIGINT)
            : f == "MIN" || f == "MAX" || f == "ANY_VALUE" || f == "FIRST" || f == "LAST" || f == "PERCENTILE_DISC" ? at
            : f == "GROUP_CONCAT" || f == "STRING_AGG" || f == "JSON_ARRAYAGG" || f == "JSON_AGG" || f == "JSON_OBJECTAGG" ? LType::VARCHAR
            : f.rfind("BIT_", 0) == 0 ? LType::BIGINT : LType::DOUBLE;
    out.names.push_back("__a" + std::to_string(j));
    out.types.push_back(t);
  }
  for (size_t gidx = 0; gidx < members.size(); ++gidx) {
    std::vector<Value> row;
    for (size_t g = 0; g < groups.size(); ++g) row.push_back(r.vals[firstRow[gidx]][g]);
    for (size_t j = 0; j < aggs.size(); ++j) {
      const ExprP& a = aggs[j];
      const std::string f = a->op;
      std::vector<size_t> rows = members[gidx];
      if (!ac[j].ord.empty())
        std::stable_sort(rows.begin(), rows.end(), [&](size_t x, size_t y) {
          for (auto& [c, d] : ac[j].ord) {
            int k = cmpVal(r.vals[x][c], r.vals[y][c]);
            if (k) return d ? k > 0 : k < 0;
          }
          return false;
        });
      // Values of the first argument (rows with a NULL argument are skipped, as in SQL).
      std::vector<Value> vals;
      std::set<std::string> seen;
      size_t starCount = rows.size();
      for (size_t rr : rows) {
        if (ac[j].args.empty()) continue;
        bool nul = false;
        for (int c : ac[j].args) nul |= r.vals[rr][c].isNull() && (f != "JSON_OBJECTAGG" || c == ac[j].args[0]);
        if (nul && f != "JSON_ARRAYAGG" && f != "JSON_AGG") continue;
        Value v = r.vals[rr][ac[j].args[0]];
        if ((f == "GROUP_CONCAT") && ac[j].args.size() > 1) {  // GROUP_CONCAT(a, b): concatenated
          std::string cat;
          for (int c : ac[j].args) cat += r.vals[rr][c].toString();
          v = Value::S(cat);
        }
        if (f == "JSON_OBJECTAGG") v = Value::S(jsonVal(Value::S(v.toString())) + ":" + jsonVal(r.vals[rr][ac[j].args[1]]));
        if (a->distinct) {
          std::string k = v.isNull() ? "\x02" : v.kind == Value::STR && ciColl() ? foldCase(v.s) : v.toString();
          if (!seen.insert(k).second) continue;
        }
        vals.push_back(v);
      }
      Value res = Value::Null();
      if (f == "COUNT") res = Value::I(ac[j].args.empty() ? (int64_t)starCount : (int64_t)vals.size());
      else if (f == "SUM" || f == "AVG") {
        if (!vals.empty()) {
          double acc = 0;
          for (auto& v : vals) acc += numOf(v);
          res = f == "AVG" ? Value::D(acc / vals.size()) : out.types[groups.size() + j] == LType::DOUBLE ? Value::D(acc) : Value::I((int64_t)std::llround(acc));
        }
      } else if (f == "MIN" || f == "MAX") {
        for (auto& v : vals)
          if (res.isNull() || (f == "MIN" ? cmpVal(v, res) < 0 : cmpVal(v, res) > 0)) res = v;
      } else if (f == "ANY_VALUE") {
        if (!vals.empty()) res = vals[0];
      } else if (f == "GROUP_CONCAT" || f == "STRING_AGG") {
        std::string sep = a->castType == "SEP" ? a->name : ",";
        if (f == "STRING_AGG" && a->args.size() > 1 && a->args[1]->kind == Expr::LIT) sep = a->args[1]->lit.toString();
        if (!vals.empty()) {
          std::string cat;
          for (size_t i = 0; i < vals.size(); ++i) cat += (i ? sep : "") + vals[i].toString();
          res = Value::S(cat);
        }
      } else if (f == "JSON_ARRAYAGG" || f == "JSON_AGG") {
        if (!vals.empty()) {
          std::string o = "[";
          const bool raw = !a->args.empty() && isJsonExpr(a->args[0], nullptr);
          for (size_t i = 0; i < vals.size(); ++i) o += (i ? ", " : "") + (raw ? jsonOfValue(vals[i], true) : jsonVal(vals[i]));
          res = Value::S(o + "]");
        }
      } else if (f == "JSON_OBJECTAGG") {
        if (!vals.empty()) {
          std::string o = "{";
          for (size_t i = 0; i < vals.size(); ++i) o += (i ? "," : "") + vals[i].s;
          res = Value::S(o + "}");
        }
      } else if (f.rfind("BIT_", 0) == 0) {
        int64_t acc = f == "BIT_AND" ? -1 : 0;
        for (auto& v : vals) {
          int64_t x = (int64_t)numOf(v);
          acc = f == "BIT_AND" ? acc & x : f == "BIT_OR" ? acc | x : acc ^ x;
        }
        res = Value::I(acc);
      } else if (f == "FIRST" || f == "LAST") {  // FIRST(v, t) / LAST(v, t): v at the smallest / largest t
        Value best, bestT;
        bool have = false;
        for (size_t rr : rows) {
          const Value& v = r.vals[rr][ac[j].args[0]];
          const Value t = ac[j].args.size() > 1 ? r.vals[rr][ac[j].args[1]] : Value::I((int64_t)rr);
          if (t.isNull()) continue;
          const int c = have ? cmpVal(t, bestT) : 0;
          if (!have || (f == "FIRST" ? c < 0 : c > 0)) best = v, bestT = t, have = true;
        }
        if (have) res = best;
      } else if (f == "PERCENTILE_CONT" || f == "PERCENTILE_DISC" || f == "APPROX_PERCENTILE") {
        if (!vals.empty()) {
          double p = 0.5;
          if (ac[j].args.size() > 1 && !rows.empty()) p = numOf(r.vals[rows[0]][ac[j].args[1]]);
          p = std::min(1.0, std::max(0.0, p));
          std::vector<double> d;
          for (auto& v : vals) d.push_back(numOf(v));
          std::sort(d.begin(), d.end());
          if (f == "PERCENTILE_DISC") {
            size_t k = (size_t)std::ceil(p * d.size());
            res = Value::D(d[k == 0 ? 0 : k - 1]);
          } else {
            const double pos = p * (d.size() - 1);
            const size_t lo = (size_t)std::floor(pos), hi = (size_t)std::ceil(pos);
            res = Value::D(d[lo] + (d[hi] - d[lo]) * (pos - lo));
          }
        }
      } else if (f == "MEDIAN") {
        if (!vals.empty()) {
          std::vector<double> d;
          for (auto& v : vals) d.push_back(numOf(v));
          std::sort(d.begin(), d.end());
          res = Value::D(d.size() % 2 ? d[d.size() / 2] : (d[d.size() / 2 - 1] + d[d.size() / 2]) / 2);
        }
      } else {  // STDDEV / VARIANCE family
        const bool samp = f == "STDDEV_SAMP" || f == "VAR_SAMP";
        if (vals.size() > (samp ? 1u : 0u)) {
          double mean = 0, m2 = 0;
          for (auto& v : vals) mean += numOf(v);
          mean /= vals.size();
          for (auto& v : vals) m2 += (numOf(v) - mean) * (numOf(v) - mean);
          double var = m2 / (vals.size() - (samp ? 1 : 0));
          res = Value::D(f.rfind("VAR", 0) == 0 ? var : std::sqrt(var));
        }
      }
      row.push_back(res);
    }
    out.vals.push_back(std::move(row));
  }
  std::string tmp = makeTemp(out, q, "hagg");
  // Outer query over the groups.
  std::function<ExprP(const ExprP&)> rw = [&](const ExprP& e) -> ExprP {
    if (!e) return e;
    for (size_t i = 0; i < groups.size(); ++i)
      if (e->str() == groups[i]->str()) return mkCol("", "__g" + std::to_string(i));
    if (e->kind == Expr::FUNC && (isAggName(e->op) || isHostAggName(e->op)) && !e->window)
      return mkCol("", "__a" + std::to_string(aggOf.at(e->str() + (e->distinct ? "!" : ""))));
    auto y = std::make_shared<Expr>(*e);
    y->bound = false;
    for (auto& a : y->args) a = rw(a);
    return y;
  };
  Stmt outer;
  outer.type = Stmt::SELECT;
  for (auto& it : s.items) outer.items.push_back({rw(it.e), it.alias.empty() ? (it.e->kind == Expr::COL ? it.e->name : it.e->str()) : it.alias});
  outer.where = rw(s.having);
  for (auto& o : s.orderBy) {
    bool alias = o.e->kind == Expr::COL && o.e->table.empty() &&
                 std::any_of(s.items.begin(), s.items.end(), [&](const SelectItem& it) { return !it.alias.empty() && strcasecmp(it.alias.c_str(), o.e->name.c_str()) == 0; });
    bool pos = o.e->kind == Expr::LIT && o.e->lit.kind == Value::INT;
    outer.orderBy.push_back({alias || pos ? o.e : rw(o.e), o.desc});
  }
  outer.limit = s.limit;
  outer.offset = s.offset;
  outer.distinct = s.distinct;
  outer.from = {TableRef{tmp, "__hagg", nullptr, 0, nullptr}};
  return query(outer, q, explain, typed);
}

thread_local bool tTypedResult = false;
thread_local bool tRawNumber = false;
thread_local bool tColumnar = false;  // a single numeric column comes back as ResultSet::numCol (IN subqueries)
thread_local bool tInsertCols = false;
thread_local int tOnlyPart = -1;
ResultSet Engine::runStatementQuery(Stmt& s, bool explain) {
  QueryCtx q;
  ResultSet r;
  try {
    r = query(s, q, explain, tTypedResult && !explain);
  } catch (...) {
    dropTemps(q);
    throw;
  }
  dropTemps(q);
  return r;
}

}  // namespace gdb
