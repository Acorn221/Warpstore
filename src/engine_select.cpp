#include "engine_core.h"

namespace gdb {
// ------------------------------------------------------------------ SELECT

// Every table a statement reads (FROM entries, derived tables, CTE bodies, subqueries, UNION arms,
// views), as catalog keys.
static void collectTables(const Stmt& s, const std::string& db, std::set<std::string>& keys, std::set<std::string> ctes, Catalog& cat, int depth);
static void collectExpr(const ExprP& e, const std::string& db, std::set<std::string>& keys, const std::set<std::string>& ctes, Catalog& cat, int depth) {
  if (!e) return;
  if (e->sub) collectTables(*e->sub, db, keys, ctes, cat, depth);
  for (auto& a : e->args) collectExpr(a, db, keys, ctes, cat, depth);
  for (auto& a : e->partBy) collectExpr(a, db, keys, ctes, cat, depth);
  for (auto& o : e->winOrder) collectExpr(o.first, db, keys, ctes, cat, depth);
}
static void collectTables(const Stmt& s, const std::string& db, std::set<std::string>& keys, std::set<std::string> ctes, Catalog& cat, int depth) {
  if (depth > 8) return;
  for (auto& [name, body] : s.ctes) {
    if (body) collectTables(*body, db, keys, ctes, cat, depth + 1);
    ctes.insert(lower(name));
  }
  for (auto& f : s.from) {
    if (f.sub) collectTables(*f.sub, db, keys, ctes, cat, depth + 1);
    else if (!f.name.empty() && !ctes.count(lower(f.name))) {
      const std::string k = f.name.find('.') != std::string::npos ? lower(f.name) : db + "." + lower(f.name);
      std::string view;
      {
        std::lock_guard<std::mutex> g(cat.mu);
        auto it = cat.views.find(k);
        if (it != cat.views.end()) view = it->second;
      }
      if (!view.empty()) {
        if (view.rfind("/*cols:", 0) == 0) view = view.substr(view.find("*/") + 2);
        try {
          collectTables(parseSql(view), k.substr(0, k.find('.')), keys, {}, cat, depth + 1);
        } catch (std::exception&) {
        }
      } else {
        keys.insert(k);
      }
    }
    collectExpr(f.on, db, keys, ctes, cat, depth + 1);
  }
  for (auto& it : s.items) collectExpr(it.e, db, keys, ctes, cat, depth + 1);
  collectExpr(s.where, db, keys, ctes, cat, depth + 1);
  collectExpr(s.having, db, keys, ctes, cat, depth + 1);
  for (auto& g : s.groupBy) collectExpr(g, db, keys, ctes, cat, depth + 1);
  for (auto& o : s.orderBy) collectExpr(o.e, db, keys, ctes, cat, depth + 1);
  for (auto& j : s.joinConds) collectExpr(j, db, keys, ctes, cat, depth + 1);
  if (s.unionNext) collectTables(*s.unionNext, db, keys, ctes, cat, depth + 1);
}

// A statement reading several tables takes all their snapshots at one moment (under the shared commit
// gate), so it sees each transaction in all of its tables or in none. (One table: lazily, as read.)
void Engine::snapshotStatement(const Stmt& s) {
  if (!tSnapScope) return;
  std::set<std::string> keys;
  collectTables(s, lower(currentDb()), keys, {}, cat_, 0);
  std::vector<std::shared_ptr<Table>> tabs;
  for (auto& k : keys) {
    if (k.rfind("information_schema.", 0) == 0) continue;
    auto t = cat_.find(k);
    if (t && t->db != "__tmp" && !t->isShadow() && !tSnapScope->shadows.count(t.get())) tabs.push_back(t);
  }
  if (tabs.size() + tSnapScope->shadows.size() < 2) return;
  std::sort(tabs.begin(), tabs.end());
  std::shared_lock<std::shared_mutex> gate(gCommitGate);
  for (auto& t : tabs) {
    std::shared_ptr<Table> sh;
    {
      FairSharedMutex::barge = true;
      std::shared_lock<FairSharedMutex> lk(t->mu);
      FairSharedMutex::barge = false;
      sh = t->shadow();
    }
    tSnapScope->bases.push_back(t);
    tSnapScope->shadows[t.get()] = sh;
  }
}

std::shared_ptr<Table> Engine::tableForRead(const std::string& name, const Stmt& s) {
  auto t = cat_.get(name);
  if (!tSnapScope || t->db == "__tmp" || t->isShadow()) return t;
  auto it = tSnapScope->shadows.find(t.get());
  if (it != tSnapScope->shadows.end()) return it->second;
  if (s.from.size() == 1 && !s.unionNext && keyLookup(*t, s.where)) return t;  // microsecond lock on the live table
  for (auto& ix : t->indexes)
    if (ix.vec && ix.algo == 0) return t;  // IVF lists live in the index map
  std::shared_ptr<Table> sh;
  {
    FairSharedMutex::barge = true;  // (never queue behind a waiting writer for this)
    std::shared_lock<FairSharedMutex> lk(t->mu);
    FairSharedMutex::barge = false;
    sh = t->shadow();
  }
  tSnapScope->bases.push_back(t);
  tSnapScope->shadows[t.get()] = sh;
  return sh;
}

ResultSet Engine::doSelect(Stmt& s, bool explain) {
  auto t0 = Clock::now();
  std::vector<std::string> notes;
  BindCtx ctx;
  for (auto& f : s.from) {
    ctx.tabs.push_back(tableForRead(f.name, s));
    ctx.aliases.push_back(f.alias);
    ctx.usingCols.push_back(f.usingCols);
  }
  // Shared locks in a stable order (writers only ever take one table lock).
  std::vector<std::shared_ptr<Table>> lockOrder = ctx.tabs;
  std::sort(lockOrder.begin(), lockOrder.end());
  lockOrder.erase(std::unique(lockOrder.begin(), lockOrder.end()), lockOrder.end());
  // A PRIMARY KEY point lookup holds the lock for microseconds: it may pass writers waiting behind
  // long analytic queries instead of queueing with them.
  FairSharedMutex::barge = false;
  if (ctx.tabs.size() == 1 && ctx.tabs[0]->pk >= 0 && s.where && s.groupBy.empty() && !s.unionNext) {
    const std::string& pkName = ctx.tabs[0]->cols[ctx.tabs[0]->pk].name;
    std::function<bool(const ExprP&)> pkEq = [&](const ExprP& e) {
      if (e->kind == Expr::BIN && e->op == "AND") return pkEq(e->args[0]) || pkEq(e->args[1]);
      if (e->kind != Expr::BIN || e->op != "=") return false;
      const ExprP& a = e->args[0];
      const ExprP& b = e->args[1];
      return (a->kind == Expr::COL && b->kind == Expr::LIT && strcasecmp(a->name.c_str(), pkName.c_str()) == 0) ||
             (b->kind == Expr::COL && a->kind == Expr::LIT && strcasecmp(b->name.c_str(), pkName.c_str()) == 0);
    };
    FairSharedMutex::barge = pkEq(s.where);
  }
  std::vector<std::shared_lock<FairSharedMutex>> locks;
  const auto tLock = Clock::now();
  for (auto& t : lockOrder) locks.emplace_back(t->mu);
  FairSharedMutex::barge = false;
  if (envSetting("TRACE") && msSince(tLock) > 20) fprintf(stderr, "[trace] waited %.1f ms for table locks\n", msSince(tLock));

  // Expand * and bind the select list.
  std::vector<SelectItem> items;
  for (auto& it : s.items) {
    if (it.e->kind == Expr::STAR) {
      if (ctx.tabs.empty()) throw SqlError("No tables used", 1096);
      for (size_t ti = 0; ti < ctx.tabs.size(); ++ti) {
        if (!it.e->table.empty() && strcasecmp(it.e->table.c_str(), ctx.aliases[ti].c_str()) != 0) continue;
        for (auto& c : ctx.tabs[ti]->cols) items.push_back({mkCol(ctx.aliases[ti], c.name), c.name});
      }
    } else {
      items.push_back(it);
    }
  }
  // Result column names of unaliased items, from the expressions as written (binding rewrites them).
  std::vector<std::string> itemNames;
  for (auto& it : items)
    itemNames.push_back(it.e->kind == Expr::COL || (it.e->kind == Expr::LIT && !it.e->name.empty()) ? it.e->name : it.e->str());
  auto resolveAlias = [&](ExprP e) -> ExprP {
    if (e->kind == Expr::LIT && e->lit.kind == Value::INT) {
      if (e->lit.i < 1 || e->lit.i > (int64_t)items.size()) throw SqlError("Unknown column '" + e->str() + "' in 'order clause'", 1054);
      return items[e->lit.i - 1].e;
    }
    if (e->kind == Expr::COL && e->table.empty())
      for (auto& it : items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), e->name.c_str()) == 0) return it.e;
    return e;
  };
  std::function<void(ExprP&)> substAliases = [&](ExprP& e) {
    if (e->kind == Expr::COL && e->table.empty()) {
      for (auto& it : items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), e->name.c_str()) == 0) { e = it.e; return; }
    }
    for (auto& a : e->args) substAliases(a);
  };
  {
    bool plain = s.groupBy.empty() && !s.distinct && s.orderBy.empty() && !s.having && s.limit >= 0 && s.offset + s.limit <= 100000;
    for (auto& it : items) plain = plain && !hasAgg(it.e);
    ctx.lazyStr = plain;
    for (auto& it : items) bind(it.e, ctx);
    ctx.lazyStr = false;
  }
  std::vector<ExprP> groupBy;
  for (auto& g : s.groupBy) {
    ExprP e = resolveAlias(g);
    bind(e, ctx);
    groupBy.push_back(e);
  }
  ExprP having = s.having;
  if (having) { substAliases(having); bind(having, ctx); }
  std::vector<OrderItem> orderBy;
  std::function<void(ExprP&)> substOrder = [&](ExprP& e) {  // select-list aliases inside ORDER BY expressions (not in aggregates)
    if (e->kind == Expr::COL && e->table.empty()) {
      for (auto& it : items)
        if (!it.alias.empty() && strcasecmp(it.alias.c_str(), e->name.c_str()) == 0) { e = it.e; return; }
      return;
    }
    if (e->kind == Expr::FUNC && (isAggName(e->op) || isHostAggName(e->op))) return;
    for (auto& a : e->args) substOrder(a);
  };
  for (auto& o : s.orderBy) {
    OrderItem oi{resolveAlias(o.e), o.desc};
    if (oi.e == o.e && !oi.e->args.empty() && !oi.e->bound) {
      oi.e = cloneExpr(oi.e);
      substOrder(oi.e);
    }
    bind(oi.e, ctx);
    orderBy.push_back(oi);
  }
  ExprP where = s.where;
  if (where) bind(where, ctx);
  std::vector<ExprP> conj;
  flattenAnd(where, conj);
  for (auto& jc : s.joinConds) {
    bind(jc, ctx);
    flattenAnd(jc, conj);
  }
  // Outer joins: which tables can come out NULL-extended; ON conditions are bound per table.
  bool anyOuter = false;
  for (size_t i = 0; i < s.from.size(); ++i) {
    const int j = s.from[i].join;
    if (j == 1) ctx.outerTabs.insert((int)i);
    else if (j == 2) for (size_t k = 0; k < i; ++k) ctx.outerTabs.insert((int)k);
    else if (j == 3) for (size_t k = 0; k <= i; ++k) ctx.outerTabs.insert((int)k);
    anyOuter |= j != 0;
    if (s.from[i].on) bind(s.from[i].on, ctx);
  }
  bool aggQuery = !groupBy.empty() || (having && hasAgg(having));
  for (auto& it : items) aggQuery |= hasAgg(it.e);
  for (auto& o : orderBy) aggQuery |= hasAgg(o.e);
  if (s.distinct && !aggQuery) {
    for (auto& it : items) groupBy.push_back(it.e);
    aggQuery = true;
  }
  if (groupBy.size() > (size_t)kMaxKeys) throw SqlError("at most 8 GROUP BY expressions are supported", 1235);
  // Case-insensitive collation: string group keys whose dictionary has case variants are grouped
  // by a canonical code ('Bob' and 'bob' form one group), everywhere the key appears.
  if (ciColl()) {
    for (auto& g : groupBy) {
      Dict* d = g->ty == LType::VARCHAR ? ctx.dict(*g) : nullptr;
      if (!d) continue;
      auto cmap = ciCanon(*d);
      if (!cmap) continue;
      const std::string orig = canon(g);
      ExprP m = boundNode(Expr::FUNC, "#MAP", {g}, LType::VARCHAR);
      m->aux = cmap;
      ctx.exprDict[m.get()] = d;
      std::function<void(ExprP&)> sub = [&](ExprP& e) {
        if (!e || e == m) return;
        if (isAgg(e)) return;
        if (canon(e) == orig) { e = m; return; }
        for (auto& a : e->args) sub(a);
      };
      for (auto& it : items) sub(it.e);
      if (having) sub(having);
      for (auto& o : orderBy) sub(o.e);
      g = m;
    }
  }

  ResultSet res;
  res.isResult = true;
  for (size_t ii = 0; ii < items.size(); ++ii) {
    auto& it = items[ii];
    // information_schema columns keep their upper-case names (MySQL 8 returns TABLE_NAME for `select table_name`)
    const bool isCol = it.e->kind == Expr::COL && it.e->bt >= 0 && it.e->bt < (int)ctx.tabs.size();
    const bool isI_S = isCol && ctx.tabs[it.e->bt]->name.rfind("__tmp", 0) == 0 && ctx.tabs[it.e->bt]->name.find("_is_") != std::string::npos;
    res.jsonCols.push_back(isJsonExpr(it.e, &ctx));
    res.names.push_back(!it.alias.empty() ? it.alias : isI_S ? ctx.tabs[it.e->bt]->cols[it.e->bc].name
                        : ii < itemNames.size() ? itemNames[ii] : it.e->kind == Expr::COL ? it.e->name : it.e->str());
    res.types.push_back(it.e->ty);
  }
  if (tDescribeOnly) return res;  // COM_STMT_PREPARE: the result columns only
  auto finish = [&](std::vector<std::vector<Value>>& rows, std::vector<std::vector<Value>>& sortKeys) {
    if (!orderBy.empty()) {
      std::vector<size_t> perm(rows.size());
      for (size_t i = 0; i < perm.size(); ++i) perm[i] = i;
      std::stable_sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        for (size_t k = 0; k < orderBy.size(); ++k) {
          int c = cmpVal(sortKeys[a][k], sortKeys[b][k]);
          if (c) return orderBy[k].desc ? c > 0 : c < 0;
        }
        return false;
      });
      std::vector<std::vector<Value>> sorted;
      sorted.reserve(rows.size());
      for (size_t i : perm) sorted.push_back(std::move(rows[i]));
      rows.swap(sorted);
    }
    int64_t start = std::min<int64_t>(s.offset, (int64_t)rows.size());
    int64_t end = s.limit < 0 ? (int64_t)rows.size() : std::min<int64_t>((int64_t)rows.size(), start + s.limit);
    for (int64_t i = start; i < end; ++i) {
      if (tWantVals) {
        res.vals.push_back(std::move(rows[i]));
        continue;
      }
      std::vector<std::string> out;
      for (auto& v : rows[i]) out.push_back(cell(v));
      res.rows.push_back(std::move(out));
    }
    res.typed = tWantVals;
  };
  auto explainResult = [&]() {
    ResultSet r;
    r.isResult = true;
    r.names = {"plan"};
    r.types = {LType::VARCHAR};
    for (auto& n : notes) r.rows.push_back({n});
    return r;
  };

  // ---- No FROM: constant expressions.
  if (ctx.tabs.empty()) {
    PostCtx pc;
    std::vector<std::vector<Value>> rows(1), keys(1);
    for (auto& it : items) rows[0].push_back(evalPost(it.e, pc));
    finish(rows, keys);
    return res;
  }

  // ---- Route 1: primary-key point lookup on the CPU (no GPU launch, microseconds).
  int64_t pkKey;
  if (ctx.tabs.size() == 1 && !aggQuery && !foreignPending(*ctx.tabs[0]) && findPkEq(conj, *ctx.tabs[0], 0, pkKey)) {
    Table& t = *ctx.tabs[0];
    notes.push_back("CPU point lookup via PRIMARY KEY hash index on " + t.name + "." + t.cols[t.pk].name + " = " +
                    std::to_string(pkKey));
    if (explain) return explainResult();
    std::vector<std::vector<Value>> rows, keys;
    auto f = t.pkIndex.find(pkKey);
    if (f != t.pkIndex.end()) {
      Loc l = f->second;
      std::vector<Value> vals;
      for (size_t c = 0; c < t.cols.size(); ++c) vals.push_back(cellValue(t, l, (int)c));
      PostCtx pc;
      pc.row = &vals;
      pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
      if (!where || isTrue(evalPost(where, pc))) {
        std::vector<Value> row;
        for (auto& it : items) row.push_back(evalPost(it.e, pc));
        rows.push_back(row);
        keys.push_back({});
      }
    }
    finish(rows, keys);
    res.info = "cpu-point-lookup";
    return res;
  }

  // ---- Route 1b: equality on a secondary index (CPU hash lookup, no GPU scan).
  int64_t ixKey;
  bool ixFound = false;
  const IndexMeta* ixm = nullptr;
  if (ctx.tabs.size() == 1 && !aggQuery && !anyOuter && !foreignPending(*ctx.tabs[0]) &&
      (ixm = findIndexEq(conj, *ctx.tabs[0], 0, ixKey, ixFound))) {
    Table& t = *ctx.tabs[0];
    notes.push_back("CPU lookup via secondary index " + ixm->name + " on " + t.name + "." + t.cols[ixm->cols[0]].name);
    if (explain) return explainResult();
    ProfStage pl("index lookup");
    std::vector<std::vector<Value>> rows, keys;
    auto it = ixFound ? ixm->map.find(ixKey) : ixm->map.end();
    if (it != ixm->map.end()) {
      for (const Loc& l : it->second) {
        std::vector<Value> vals;
        for (size_t c = 0; c < t.cols.size(); ++c) vals.push_back(cellValue(t, l, (int)c));
        PostCtx pc;
        pc.row = &vals;
        pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
        if (where && !isTrue(evalPost(where, pc))) continue;
        std::vector<Value> row, key;
        for (auto& itm : items) row.push_back(evalPost(itm.e, pc));
        for (auto& o : orderBy) key.push_back(evalPost(o.e, pc));
        rows.push_back(std::move(row));
        keys.push_back(std::move(key));
      }
    }
    finish(rows, keys);
    pl.done((int64_t)res.rows.size() + (int64_t)res.vals.size(), ixm->name);
    res.info = "cpu-index-lookup";
    return res;
  }

  // ---- Route 1b: ORDER BY v <*> q DESC / v <-> q LIMIT k on a CAGRA VECTOR INDEX: the GPU graph of
  // every large VRAM-resident segment is searched in one call (deleted rows masked out), the write
  // buffers and other segments exactly; candidates are re-scored exactly, then filtered by WHERE.
  if (ctx.tabs.size() == 1 && !aggQuery && !anyOuter && orderBy.size() == 1 && s.limit >= 0 && s.limit + s.offset <= 1024 &&
      orderBy[0].e->kind == Expr::FUNC && (orderBy[0].e->op == "#VDOT" || orderBy[0].e->op == "#VEUC") &&
      orderBy[0].e->args[0]->kind == Expr::COL && (orderBy[0].e->op == "#VDOT") == orderBy[0].desc && tAnn &&
      !foreignPending(*ctx.tabs[0])) {
    Table& t = *ctx.tabs[0];
    const ExprP& oe = orderBy[0].e;
    const IndexMeta* vix = nullptr;
    for (auto& ix : t.indexes)
      if (ix.vec && ix.algo == 1 && ix.cagra && ix.cols[0] == oe->args[0]->bc) vix = &ix;
    const std::vector<double>& qd = *oe->aux;
    if (vix && (int)qd.size() == vix->dim) {
      const bool dot = oe->op == "#VDOT";
      const int d = vix->dim, vc = vix->cols[0];
      const size_t want = (size_t)(s.limit + s.offset);
      // WHERE the GPU can evaluate becomes a per-segment keep-bitset (filtered graph search);
      // otherwise candidates are over-fetched and filtered on the host.
      const bool gpuWhere = where && gpuCompilable(where, ctx);
      Compiler wcomp(&ctx);
      Prog wprog;
      if (gpuWhere) {
        wprog = wcomp.compile(where);
        wcomp.uploadSets();
      }
      const int k = (int)std::min<size_t>(1024, where && !gpuWhere ? std::max<size_t>(want * 8, 64) : want);
      const int itopk = std::max(k, tCagraItopk > 0 ? tCagraItopk : vix->itopk > 0 ? vix->itopk : 64);
      std::vector<float> q(qd.begin(), qd.end());
      ProfStage pl("vector index");
      // Graphs for the segments that have (or can get) one; everything else is scored exactly.
      std::vector<CagraGraph*> graphs;
      std::vector<int> graphSeg;
      std::vector<const uint32_t*> keep;
      std::vector<std::shared_ptr<DevVec<uint32_t>>> keepBufs;
      std::vector<int> exactSegs;
      std::vector<std::shared_ptr<CagraGraph>> hold;
      std::vector<std::pair<int, std::vector<uint32_t>>> selRows;  // gpuWhere: per graph segment, rows passing WHERE
      int64_t selected = 0;
      int built = 0;
      for (size_t si = 0; si < t.segs.size(); ++si) {
        Segment& sg = *t.segs[si];
        if (sg.n == 0 || sg.ndel() == sg.n) continue;
        std::shared_ptr<CagraGraph> g;
        if (segCold(*sg.d, vc)) segLoad(*sg.d, vc);  // (graph columns stay in memory)
        if (sg.n >= 2048 && sg.dev[vc] && !sg.mapped[vc]) {
          std::lock_guard<std::mutex> lk(vix->cagra->mu);
          auto it = vix->cagra->graphs.find(sg.id);
          if (it != vix->cagra->graphs.end()) g = it->second;
          else if (!vix->cagra->failed.count(sg.id)) {
            std::string err;
            g = cagraBuild((const float*)sg.dev[vc], sg.n, d, dot, vix->graphDegree, err);
            if (g) vix->cagra->graphs[sg.id] = g, ++built;
            else {
              vix->cagra->failed.insert(sg.id);
              fprintf(stderr, "[warpstore] CAGRA build failed for %s.%s: %s (exact search used)\n", t.name.c_str(), vix->name.c_str(), err.c_str());
            }
          }
        }
        if (!g) {
          exactSegs.push_back((int)si);
          continue;
        }
        if (gpuWhere) {  // rows passing WHERE (and not deleted) -> keep-bitset
          const uint32_t* sdel = sg.delDev();
          const BatchRef br{sg.part, sg.n, &sg.dev, sdel, &sg.enc, 0, &sg};
          struct Temps { ~Temps() { releaseScanTemps(); } } temps;
          for (size_t kk = 0; kk < wcomp.slotCols.size(); ++kk) bindView(wcomp.env.cols[kk], br, wcomp.slotCols[kk].second);
          scanFlush();
          DevVec<int32_t> idx(sg.n);
          const int64_t m = gpuSelect(wcomp.env, true, wprog, sg.n, sdel, idx.p);
          if (m == 0) continue;
          std::vector<uint32_t> bits((sg.n + 31) / 32, 0);
          for (int32_t r : idx.download(m)) bits[r >> 5] |= 1u << (r & 31);
          selected += m;
          selRows.push_back({(int)si, std::move(bits)});
          hold.push_back(g);
          graphs.push_back(g.get());
          graphSeg.push_back((int)si);
          auto b = std::make_shared<DevVec<uint32_t>>((int64_t)selRows.back().second.size());
          h2d(b->p, selRows.back().second.data(), selRows.back().second.size() * 4);
          keepBufs.push_back(b);
          keep.push_back(b->p);
          continue;
        }
        hold.push_back(g);
        graphs.push_back(g.get());
        graphSeg.push_back((int)si);
        if (sg.ndel() > 0) {  // keep-bitset: the complement of the deleted-row bitmap
          std::vector<uint32_t> bits(sg.delHost().size());
          for (size_t w = 0; w < bits.size(); ++w) bits[w] = ~sg.delHost()[w];
          auto b = std::make_shared<DevVec<uint32_t>>((int64_t)bits.size());
          h2d(b->p, bits.data(), bits.size() * 4);
          keepBufs.push_back(b);
          keep.push_back(b->p);
        } else {
          keep.push_back(nullptr);
        }
      }
      notes.push_back("ANN via VECTOR INDEX " + vix->name + " (CAGRA GPU graph: " + std::to_string(graphs.size()) + " segment graphs" +
                      (built ? " (" + std::to_string(built) + " built now)" : "") + ", itopk " + std::to_string(itopk) + ", " +
                      std::to_string(exactSegs.size()) + " segments + write buffers exact) on " + t.name + "." + t.cols[vc].name);
      if (explain) return explainResult();
      std::vector<CagraHit> hits;
      std::string err;
      // A selective filter leaves few rows: scoring them exactly beats a graph walk that must skip
      // most of its nodes.
      const bool exactSelected = gpuWhere && selected < 8192;
      if (exactSelected) notes.push_back("WHERE keeps " + std::to_string(selected) + " rows: scored exactly");
      if (exactSelected || cagraSearch(graphs, keep, q.data(), d, k, itopk, hits, err)) {
        auto score = [&](const float* v) {
          double acc = 0;
          for (int i = 0; i < d; ++i) acc += dot ? (double)v[i] * qd[i] : ((double)v[i] - qd[i]) * ((double)v[i] - qd[i]);
          return dot ? -acc : acc;  // lower is better
        };
        std::vector<std::pair<double, Loc>> cand;
        if (exactSelected) {
          for (auto& [si, bits] : selRows)
            for (size_t w = 0; w < bits.size(); ++w)
              for (uint32_t m = bits[w]; m; m &= m - 1) {
                const Loc l{si, (int32_t)(w * 32 + __builtin_ctz(m))};
                cand.push_back({score((const float*)t.colData(l, vc).data() + (size_t)l.row * d), l});
              }
        }
        for (auto& h : hits) {
          const Loc l{graphSeg[h.graph], (int32_t)h.row};
          cand.push_back({score((const float*)t.colData(l, vc).data() + (size_t)l.row * d), l});
        }
        auto exactRows = [&](Loc first, int64_t n, const DelBits& del) {
          for (int64_t r = 0; r < n; ++r) {
            if (del.test(r)) continue;
            const Loc l{first.seg, (int32_t)r};
            cand.push_back({score((const float*)t.colData(l, vc).data() + (size_t)r * d), l});
          }
        };
        for (int si : exactSegs) exactRows(Loc{si, 0}, t.segs[si]->n, *t.segs[si]->del);
        for (int p = 0; p < t.nparts; ++p) exactRows(bufLoc(p, 0), t.bufs[p]->n, *t.bufs[p]->del);
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first < b.first; });
        std::vector<std::vector<Value>> rows, keys;
        for (auto& [sc, l] : cand) {
          if (rows.size() >= want) break;
          std::vector<Value> vals;
          for (size_t c = 0; c < t.cols.size(); ++c) vals.push_back(cellValue(t, l, (int)c));
          PostCtx pc;
          pc.row = &vals;
          pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
          if (where && !isTrue(evalPost(where, pc))) continue;
          std::vector<Value> row;
          for (auto& itm : items) row.push_back(evalPost(itm.e, pc));
          rows.push_back(std::move(row));
          keys.push_back({Value::D(dot ? -sc : std::sqrt(sc))});
        }
        if (rows.size() >= want || !where) {  // a selective WHERE that left too few: exact path below
          finish(rows, keys);
          pl.done((int64_t)cand.size(), vix->name);
          res.info = "ann-cagra";
          return res;
        }
        notes.push_back("CAGRA candidates too few after WHERE: exact search");
      } else {
        fprintf(stderr, "[warpstore] CAGRA search failed: %s (exact search used)\n", err.c_str());
      }
    }
  }

  // ---- Route 1c: ORDER BY v <*> q DESC / v <-> q LIMIT k on a VECTOR INDEX: probe the nprobe
  // nearest IVF lists on the CPU, score only their vectors, then evaluate the select list for the best.
  if (ctx.tabs.size() == 1 && !aggQuery && !anyOuter && orderBy.size() == 1 && s.limit >= 0 && s.limit <= 100000 &&
      orderBy[0].e->kind == Expr::FUNC && (orderBy[0].e->op == "#VDOT" || orderBy[0].e->op == "#VEUC") &&
      orderBy[0].e->args[0]->kind == Expr::COL && (orderBy[0].e->op == "#VDOT") == orderBy[0].desc && tAnn &&
      !foreignPending(*ctx.tabs[0])) {
    Table& t = *ctx.tabs[0];
    const ExprP& oe = orderBy[0].e;
    const IndexMeta* vix = nullptr;
    for (auto& ix : t.indexes)
      if (ix.vec && ix.nlist > 0 && ix.cols[0] == oe->args[0]->bc) vix = &ix;
    if (vix) {
      const bool dot = oe->op == "#VDOT";
      const int d = vix->dim;
      const std::vector<double>& q = *oe->aux;
      if ((int)q.size() == d) {
        int nprobe = tNprobe > 0 ? tNprobe : vix->nprobe > 0 ? vix->nprobe : std::max(8, vix->nlist / 32);
        nprobe = std::min(nprobe, vix->nlist);
        notes.push_back("ANN via VECTOR INDEX " + vix->name + " (IVF, " + std::to_string(vix->nlist) + " lists, nprobe " +
                        std::to_string(nprobe) + ") on " + t.name + "." + t.cols[vix->cols[0]].name);
        if (explain) return explainResult();
        ProfStage pl("vector index");
        auto score = [&](const float* v) {
          double acc = 0;
          for (int i = 0; i < d; ++i) acc += dot ? v[i] * q[i] : (v[i] - q[i]) * (v[i] - q[i]);
          return dot ? -acc : acc;  // lower is better
        };
        std::vector<std::pair<double, int>> cs(vix->nlist);
        for (int c = 0; c < vix->nlist; ++c) cs[c] = {score(vix->cent.data() + (size_t)c * d), c};
        std::partial_sort(cs.begin(), cs.begin() + nprobe, cs.end());
        std::vector<std::pair<double, Loc>> cand;
        for (int i = 0; i < nprobe; ++i) {
          auto it = vix->map.find(cs[i].second);
          if (it == vix->map.end()) continue;
          for (const Loc& l : it->second)
            cand.push_back({score((const float*)t.colData(l, vix->cols[0]).data() + (size_t)l.row * d), l});
        }
        // Rows not yet assigned to the trained lists (key 0 before training) are covered by the lists.
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first < b.first; });
        std::vector<std::vector<Value>> rows, keys;
        const size_t want = (size_t)(s.limit + s.offset);
        for (auto& [sc, l] : cand) {
          if (rows.size() >= want) break;
          std::vector<Value> vals;
          for (size_t c = 0; c < t.cols.size(); ++c) vals.push_back(cellValue(t, l, (int)c));
          PostCtx pc;
          pc.row = &vals;
          pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
          if (where && !isTrue(evalPost(where, pc))) continue;
          std::vector<Value> row;
          for (auto& itm : items) row.push_back(evalPost(itm.e, pc));
          rows.push_back(std::move(row));
          keys.push_back({Value::D(dot ? -sc : std::sqrt(sc))});
        }
        finish(rows, keys);
        pl.done((int64_t)cand.size(), vix->name);
        res.info = "ann-ivf";
        return res;
      }
    }
  }

  // ---- Split predicates: per-table pushdown, equi-join edges, residual.
  std::vector<std::vector<ExprP>> pushdown(ctx.tabs.size());
  struct Edge { int at, ac, bt, bc; ExprP e; };
  std::vector<Edge> edges;
  std::vector<ExprP> residual;
  for (auto& p : conj) {
    std::set<int> ts;
    tablesOf(p, ts);
    bool touchesOuter = false;
    for (int t : ts) touchesOuter |= ctx.outerTabs.count(t) > 0;
    if (touchesOuter) {  // WHERE on a NULL-extended table filters after the outer join
      residual.push_back(p);
    } else if (ts.size() <= 1) {
      pushdown[ts.empty() ? 0 : *ts.begin()].push_back(p);
    } else if (ts.size() == 2 && p->kind == Expr::BIN && p->op == "=" && p->args[0]->kind == Expr::COL &&
               p->args[1]->kind == Expr::COL &&
               (p->args[0]->ty != LType::VARCHAR || (p->args[1]->ty == LType::VARCHAR && ctx.dict(*p->args[0]) && ctx.dict(*p->args[1])))) {
      edges.push_back({p->args[0]->bt, p->args[0]->bc, p->args[1]->bt, p->args[1]->bc, p});
    } else {
      residual.push_back(p);
    }
  }

  // Aggregate slots: slot 0 is a hidden COUNT(*) used for COUNT/AVG and empty-group detection.
  std::map<std::string, AggRef> aggs;
  AggPlan plan;
  std::vector<ExprP> aggArgs;  // per slot (null for COUNT)
  plan.naggs = 1;
  plan.ops[0] = AGG_COUNT;
  aggArgs.push_back(nullptr);
  std::map<std::string, int> slotByCanon;
  // Slot counting the non-NULL rows of an aggregate argument (0 = COUNT(*) when it can't be NULL).
  auto countSlot = [&](const ExprP& x) -> int {
    if (x->kind == Expr::STAR || !nullable(x, ctx)) return 0;
    std::string sk = "cnt" + canon(x);
    auto it = slotByCanon.find(sk);
    if (it != slotByCanon.end()) return it->second;
    if (plan.naggs >= kMaxAggs) throw SqlError("too many aggregates", 1105);
    plan.ops[plan.naggs] = AGG_SUM;
    aggArgs.push_back(boundNode(Expr::FUNC, "ISNOTNULL", {x}, LType::INT));
    slotByCanon.emplace(sk, plan.naggs);
    return plan.naggs++;
  };
  ExprP distX;  // COUNT(DISTINCT x): computed by the two-phase GPU path below
  bool anyDistinct = false;  // (grouped by x in its first phase, so BIGINT sums split as with GROUP BY)
  std::function<void(const ExprP&)> findDistinct = [&](const ExprP& e) {
    if (!e) return;
    if (isAgg(e) && e->distinct) anyDistinct = true;
    for (auto& a : e->args) findDistinct(a);
  };
  for (auto& it : items) findDistinct(it.e);
  findDistinct(having);
  for (auto& o : orderBy) findDistinct(o.e);
  std::function<void(const ExprP&)> collectAggs = [&](const ExprP& e) {
    if (isAgg(e)) {
      if (e->distinct) {
        if (distX && canon(distX) != canon(e->args[0])) throw SqlError("internal: several DISTINCT arguments", 1105);
        distX = e->args[0];
      }
      std::string k = canon(e);
      if (aggs.count(k)) return;
      AggRef r;
      r.distinct = e->distinct;
      r.fn = e->op;
      r.argTy = e->args[0]->ty;
      r.dict = ctx.dict(*e);
      if (auto u = ctx.unrank.find(e.get()); u != ctx.unrank.end()) r.unrank = u->second;
      if (r.dict) r.argTy = LType::VARCHAR;
      r.scale = e->scale;
      r.cntSlot = countSlot(e->args[0]);
      if (r.fn == "COUNT") r.slot = r.cntSlot;
      if (r.fn != "COUNT") {
        uint8_t op = r.fn == "MIN" ? AGG_MIN : r.fn == "MAX" ? AGG_MAX : AGG_SUM;
        const ExprP& arg = e->args[0];
        const bool rawDec = op == AGG_SUM && arg->kind == Expr::COL && arg->ty == LType::DECIMAL;
        if (rawDec) r.decScale = arg->scale;
        std::string sk = std::to_string(op) + (rawDec ? "raw" : "") + canon(e->args[0]);
        auto it = slotByCanon.find(sk);
        if (it == slotByCanon.end()) {
          if (plan.naggs >= kMaxAggs) throw SqlError("too many aggregates", 1105);
          plan.ops[plan.naggs] = op;
          aggArgs.push_back(rawDec ? boundNode(Expr::FUNC, "#RAW", {arg}, LType::BIGINT) : e->args[0]);
          it = slotByCanon.emplace(sk, plan.naggs++).first;
          // Grouped SUM of a 64-bit integer column: exact as two int64 sums of its 32-bit halves.
          if (op == AGG_SUM && (!groupBy.empty() || anyDistinct) && arg->kind == Expr::COL && (arg->ty == LType::BIGINT || arg->ty == LType::DATETIME)) {
            if (plan.naggs >= kMaxAggs) throw SqlError("too many aggregates", 1105);
            plan.ops[plan.naggs] = AGG_SUM;
            plan.aggPart[it->second] = 1;
            plan.aggPart[plan.naggs] = 2;
            aggArgs.push_back(arg);
            slotByCanon.emplace(sk + "#hi", plan.naggs++);
          }
        }
        r.slot = it->second;
        if (auto h = slotByCanon.find(sk + "#hi"); h != slotByCanon.end()) r.hiSlot = h->second;
      }
      aggs[k] = r;
      return;
    }
    for (auto& a : e->args) collectAggs(a);
  };

  // Columns needed per table (for materialization paths).
  std::set<std::pair<int, int>> needed;
  for (auto& it : items) colsOf(it.e, needed);
  for (auto& g : groupBy) colsOf(g, needed);
  for (auto& o : orderBy) colsOf(o.e, needed);
  if (having) colsOf(having, needed);
  for (auto& r : residual) colsOf(r, needed);
  for (auto& e : edges) colsOf(e.e, needed);
  for (auto& f : s.from)
    if (f.on) colsOf(f.on, needed);
  auto needOf = [&](int bt) {
    std::set<int> c;
    for (auto& [t, col] : needed)
      if (t == bt) c.insert(col);
    if (c.empty()) c.insert(0);
    return c;
  };

  // Build the joined relation (multi-table). Small tables are filtered up front. Big ones are
  // scanned only when the join order reaches them, with join filters: exact bitmaps of the join
  // keys already in the joined relation, applied inside the big table's scan (sideways
  // information passing), so rows that cannot match are never gathered or joined.
  // With outer joins the tables are joined in FROM order (join order matters for their semantics).
  // Equality between expressions of the two sides (a.x = b.y + 1, a.d = DATE(b.ts), ..): both key
  // expressions evaluated on the GPU into temporary key columns, then a hash join on them.
  auto exprEquiCond = [&](const std::set<int>& lt, const std::set<int>& rt, const std::vector<ExprP>& conds, ExprP& le, ExprP& re) {
    for (auto& c : conds) {
      if (c->kind != Expr::BIN || c->op != "=" || hasAgg(c)) continue;
      std::set<int> t0, t1;
      tablesOf(c->args[0], t0);
      tablesOf(c->args[1], t1);
      if (t0.empty() || t1.empty()) continue;
      auto within = [](const std::set<int>& a, const std::set<int>& b) { return std::includes(b.begin(), b.end(), a.begin(), a.end()); };
      if (c->args[0]->ty == LType::VARCHAR || c->args[1]->ty == LType::VARCHAR) continue;
      if (within(t0, lt) && within(t1, rt)) { le = c->args[0], re = c->args[1]; return true; }
      if (within(t1, lt) && within(t0, rt)) { le = c->args[1], re = c->args[0]; return true; }
    }
    return false;
  };
  auto addKeyCol = [&](Rel& r, const ExprP& e, int tag, bool integral) {
    Compiler comp(&ctx);
    Prog pr = comp.compile(e);
    comp.uploadSets();
    bindRelCols(comp, r);
    DevVec<double> v(std::max<int64_t>(r.n, 1));
    gpuEvalAt(comp.env, pr, nullptr, r.n, v.p);
    std::vector<double> h = v.download(r.n);
    std::vector<int64_t> k(r.n);
    for (int64_t i = 0; i < r.n; ++i) {
      double d = h[i];
      if (d != d) { k[i] = kNullI64; continue; }
      if (integral) { k[i] = std::llround(d); continue; }
      if (d == 0) d = 0;  // -0.0 = 0.0
      memcpy(&k[i], &d, 8);
    }
    RelCol c{tag, 0, PType::I64, DevVec<uint8_t>(std::max<int64_t>(r.n, 1) * 8), 8};
    if (r.n) h2d(c.data.p, k.data(), r.n * 8);
    r.cols.push_back(std::move(c));
  };
  auto exprEquiJoin = [&](Rel& left, Rel& right, const std::vector<ExprP>& conds, Rel& out) {
    ExprP le, re;
    if (!exprEquiCond(left.tabs, right.tabs, conds, le, re)) return false;
    const int kA = -2000, kB = -2001;
    auto intTy = [](LType t) { return t == LType::INT || t == LType::BIGINT || t == LType::DATE || t == LType::DATETIME; };
    const bool integral = intTy(le->ty) && intTy(re->ty);  // else both as float64 bit patterns
    addKeyCol(left, le, kA, integral);
    addKeyCol(right, re, kB, integral);
    out = hashJoin(left, right, kA, 0, kB, 0, notes);
    for (auto* r : {&out, &left, &right})
      r->cols.erase(std::remove_if(r->cols.begin(), r->cols.end(), [&](const RelCol& c) { return c.bt == kA || c.bt == kB; }), r->cols.end());
    notes.push_back("    (hash join on expression keys: " + le->str() + " = " + re->str() + ")");
    return true;
  };

  // Joins tables [from, end) onto cur in FROM order (outer joins: the order matters).
  auto buildOrdered = [&](Rel cur, size_t from, std::vector<bool> edgeUsed) -> Rel {
    for (size_t i = from; i < ctx.tabs.size(); ++i) {
      const TableRef& ref = s.from[i];
      if (ref.join == 0) {
        Rel ri = materialize(ctx, (int)i, pushdown[i], needOf((int)i), -1, notes);
        int ei = -1;
        for (size_t k = 0; k < edges.size() && ei < 0; ++k) {
          auto& e = edges[k];
          if (!edgeUsed[k] && ((cur.tabs.count(e.at) && e.bt == (int)i) || (cur.tabs.count(e.bt) && e.at == (int)i))) ei = (int)k;
        }
        if (ei < 0) {
          Rel j;
          if (exprEquiJoin(cur, ri, residual, j)) cur = std::move(j);
          else cur = crossJoin(cur, ri, notes);
        } else {
          edgeUsed[ei] = true;
          auto& e = edges[ei];
          const Dict* da = ctx.tabs[e.at]->cols[e.ac].lt == LType::VARCHAR ? ctx.tabs[e.at]->dicts[e.ac].get() : nullptr;
          const Dict* db = ctx.tabs[e.bt]->cols[e.bc].lt == LType::VARCHAR ? ctx.tabs[e.bt]->dicts[e.bc].get() : nullptr;
          cur = cur.tabs.count(e.at) ? hashJoin(cur, ri, e.at, e.ac, e.bt, e.bc, notes, da, db) : hashJoin(cur, ri, e.bt, e.bc, e.at, e.ac, notes, db, da);
        }
        continue;
      }
      std::vector<ExprP> on, rightOnly, other;
      flattenAnd(ref.on, on);
      int ct = -1, cc = -1, rc = -1;
      for (auto& c : on) {
        std::set<int> ts;
        tablesOf(c, ts);
        if (ct < 0 && c->kind == Expr::BIN && c->op == "=" && c->args[0]->kind == Expr::COL && c->args[1]->kind == Expr::COL &&
            (c->args[0]->ty == LType::VARCHAR) == (c->args[1]->ty == LType::VARCHAR) &&
            c->args[0]->ty != LType::DOUBLE && c->args[1]->ty != LType::DOUBLE) {  // (DOUBLE keys: the general path)
          const ExprP &a = c->args[0], &b = c->args[1];
          if (cur.tabs.count(a->bt) && b->bt == (int)i) { ct = a->bt; cc = a->bc; rc = b->bc; continue; }
          if (cur.tabs.count(b->bt) && a->bt == (int)i) { ct = b->bt; cc = b->bc; rc = a->bc; continue; }
        }
        if (ts.size() == 1 && *ts.begin() == (int)i) rightOnly.push_back(c);
        else other.push_back(c);
      }
      if (ct < 0 || !other.empty() || (ref.join != 1 && !rightOnly.empty())) {
        // General ON (no equality, or conditions beyond one equality plus right-side filters): inner
        // join on the equality (else cross join), the rest of ON as a GPU filter over the pairs, then
        // the preserved side's unmatched rows NULL-extended.
        std::vector<ExprP> rest = other;
        rest.insert(rest.end(), rightOnly.begin(), rightOnly.end());
        Rel ri = materialize(ctx, (int)i, {}, needOf((int)i), -1, notes);
        const int kIdA = -1000, kIdB = -1001;
        auto addId = [](Rel& r, int tag) {
          std::vector<int32_t> seq((size_t)r.n);
          for (int64_t k = 0; k < r.n; ++k) seq[k] = (int32_t)k;
          RelCol c{tag, 0, PType::I32, DevVec<uint8_t>(std::max<int64_t>(r.n, 1) * 4), 4};
          if (r.n) h2d(c.data.p, seq.data(), r.n * 4);
          r.cols.push_back(std::move(c));
        };
        addId(cur, kIdA);
        addId(ri, kIdB);
        Rel inner;
        if (ct >= 0) {
          const Dict* dl = ctx.tabs[ct]->cols[cc].lt == LType::VARCHAR ? ctx.tabs[ct]->dicts[cc].get() : nullptr;
          const Dict* dr = ctx.tabs[i]->cols[rc].lt == LType::VARCHAR ? ctx.tabs[i]->dicts[rc].get() : nullptr;
          inner = hashJoin(cur, ri, ct, cc, (int)i, rc, notes, dl, dr);
        } else if (exprEquiJoin(cur, ri, rest, inner)) {
        } else {
          if ((double)cur.n * (double)ri.n > 4e8)
            throw SqlError("outer join without an equality would compare " + std::to_string(cur.n) + " x " + std::to_string(ri.n) + " rows; add an equality to ON", 1235);
          inner = crossJoin(cur, ri, notes);
        }
        std::vector<int32_t> pa, pb;  // matching (left row, right row) pairs
        {
          DevVec<int32_t> sel;
          int64_t m = inner.n;
          if (!rest.empty() && inner.n > 0) {
            Compiler comp(&ctx);
            Prog fp = comp.compile(andAll(rest));
            comp.uploadSets();
            bindRelCols(comp, inner);
            sel = DevVec<int32_t>(inner.n);
            m = gpuSelect(comp.env, true, fp, inner.n, nullptr, sel.p);
          }
          const std::pair<int, std::vector<int32_t>*> idCols[] = {{kIdA, &pa}, {kIdB, &pb}};
          for (auto [tag, out] : idCols) {
            const RelCol& c = inner.cols[inner.find(tag, 0)];
            if (sel.p) {
              DevVec<int32_t> g(std::max<int64_t>(m, 1));
              gpuGather(c.data.p, PType::I32, sel.p, m, g.p);
              *out = g.download(m);
            } else {
              out->resize(m);
              if (m) d2h(out->data(), c.data.p, m * 4);
            }
          }
        }
        std::vector<uint8_t> hitA(cur.n, 0), hitB(ri.n, 0);
        for (size_t k = 0; k < pa.size(); ++k) hitA[pa[k]] = hitB[pb[k]] = 1;
        std::vector<int32_t> fa = pa, fb = pb;
        if (ref.join == 1 || ref.join == 3)
          for (int64_t k = 0; k < cur.n; ++k)
            if (!hitA[k]) fa.push_back((int32_t)k), fb.push_back(-1);
        if (ref.join == 2 || ref.join == 3)
          for (int64_t k = 0; k < ri.n; ++k)
            if (!hitB[k]) fa.push_back(-1), fb.push_back((int32_t)k);
        const int64_t m = (int64_t)fa.size();
        DevVec<int32_t> da(std::max<int64_t>(m, 1)), db(std::max<int64_t>(m, 1));
        if (m) h2d(da.p, fa.data(), m * 4), h2d(db.p, fb.data(), m * 4);
        Rel out;
        out.n = m;
        out.tabs = cur.tabs;
        out.tabs.insert(ri.tabs.begin(), ri.tabs.end());
        for (auto* side : {&cur, &ri})
          for (auto& c : side->cols) {
            if (c.bt <= kIdA) continue;
            RelCol rc2{c.bt, c.bc, c.pt, DevVec<uint8_t>(std::max<int64_t>(m, 1) * c.esize), c.esize};
            gatherCol(c.data.p, c.pt, c.esize, side == &cur ? da.p : db.p, m, rc2.data.p, true);
            out.cols.push_back(std::move(rc2));
          }
        static const char* kind[] = {"", "LEFT", "RIGHT", "FULL"};
        notes.push_back(std::string("  GPU ") + kind[ref.join] + " OUTER join on a general ON condition: " + std::to_string(pa.size()) +
                        " matching pairs -> " + std::to_string(m) + " rows");
        cur = std::move(out);
        continue;
      }
      Rel ri = materialize(ctx, (int)i, ref.join == 1 ? rightOnly : std::vector<ExprP>{}, needOf((int)i), -1, notes);
      const Dict* dl = ctx.tabs[ct]->cols[cc].lt == LType::VARCHAR ? ctx.tabs[ct]->dicts[cc].get() : nullptr;
      const Dict* dr = ctx.tabs[i]->cols[rc].lt == LType::VARCHAR ? ctx.tabs[i]->dicts[rc].get() : nullptr;
      cur = outerJoin(cur, ri, ct, cc, (int)i, rc, ref.join, notes, dl, dr);
    }
    for (size_t k = 0; k < edges.size(); ++k)
      if (!edgeUsed[k]) residual.push_back(edges[k].e);
    return cur;
  };

  auto buildJoin = [&]() -> Rel {
    // With outer joins only at the end of FROM (e.g. decorrelated subqueries), the inner-join prefix
    // is ordered by cost and the outer joins follow in order; other outer joins: FROM order.
    size_t prefixN = ctx.tabs.size();
    if (anyOuter) {
      size_t k = 0;
      while (k < s.from.size() && s.from[k].join == 0) ++k;
      bool trailing = k >= 2;
      for (size_t i = k; i < s.from.size(); ++i) trailing = trailing && s.from[i].join != 0;
      if (!trailing) return buildOrdered(materialize(ctx, 0, pushdown[0], needOf(0), -1, notes), 1, std::vector<bool>(edges.size(), false));
      prefixN = k;
    }
    const size_t nt = ctx.tabs.size();
    std::vector<Rel> rels(nt);
    std::vector<bool> mat(nt, false);
    for (size_t i = 0; i < prefixN; ++i)
      if (!tJoinFilters || ctx.tabs[i]->liveRows() < tLazyRows) {
        rels[i] = materialize(ctx, (int)i, pushdown[i], needOf((int)i), -1, notes);
        mat[i] = true;
      }
    if (std::find(mat.begin(), mat.end(), true) == mat.end()) {
      size_t sm = 0;
      for (size_t i = 1; i < prefixN; ++i)
        if (ctx.tabs[i]->liveRows() < ctx.tabs[sm]->liveRows()) sm = i;
      rels[sm] = materialize(ctx, (int)sm, pushdown[sm], needOf((int)sm), -1, notes);
      mat[sm] = true;
    }
    auto est = [&](size_t i) { return mat[i] ? (double)rels[i].n : (double)ctx.tabs[i]->liveRows(); };
    // Greedy left-deep join order. Start from the smallest relation; at each step pick the edge
    // with the smallest estimated output, using PRIMARY KEY uniqueness to recognise FK->PK joins
    // (which never grow the input). Build side is always the smaller input.
    std::vector<bool> used(nt, false);
    for (size_t i = prefixN; i < nt; ++i) used[i] = true;  // (joined afterwards, in order)
    size_t start = nt;
    for (size_t i = 0; i < nt; ++i)
      if (mat[i] && (start == nt || rels[i].n < rels[start].n)) start = i;
    Rel cur = std::move(rels[start]);
    used[start] = true;
    std::set<std::pair<int, int>> uniq;  // columns known to be unique in `cur`
    if (ctx.tabs[start]->pk >= 0) uniq.insert({(int)start, ctx.tabs[start]->pk});
    std::vector<bool> edgeUsed(edges.size(), false);
    for (size_t step = 1; step < prefixN; ++step) {
      int best = -1, bestEdge = -1;
      double bestEst = 0;
      for (size_t ei = 0; ei < edges.size(); ++ei) {
        auto& e = edges[ei];
        int other = cur.tabs.count(e.at) && !cur.tabs.count(e.bt) ? e.bt : cur.tabs.count(e.bt) && !cur.tabs.count(e.at) ? e.at : -1;
        if (other < 0 || used[other]) continue;
        bool aIsCur = other == e.bt;
        std::pair<int, int> curCol = aIsCur ? std::make_pair(e.at, e.ac) : std::make_pair(e.bt, e.bc);
        int otherCol = aIsCur ? e.bc : e.ac;
        // Without key information: rows x rows / the larger number of distinct join values (from
        // dictionary sizes / value ranges; a many-to-many join on a low-cardinality column is huge).
        auto ndv = [&](int bt, int bc, double rows) {
          int64_t lo, hi;
          double n = colRange(*ctx.tabs[bt], bc, lo, hi) ? (double)span64(lo, hi) + 1 : rows;
          return std::max(1.0, std::min(n, rows));
        };
        const double nd = std::max(ndv(curCol.first, curCol.second, (double)cur.n), ndv(other, otherCol, (double)ctx.tabs[other]->liveRows()));
        double est1 = otherCol == ctx.tabs[other]->pk ? (double)cur.n
                    : uniq.count(curCol) ? est(other) : (double)cur.n * est(other) / nd;
        if (best < 0 || est1 < bestEst || (est1 == bestEst && est(other) < est(best)))
          best = other, bestEdge = (int)ei, bestEst = est1;
      }
      if (best < 0) {  // no column equality to any remaining table: an expression equality, else a cross join
        ExprP le, re;
        for (size_t i = 0; i < nt && best < 0; ++i)
          if (!used[i] && exprEquiCond(cur.tabs, std::set<int>{(int)i}, residual, le, re)) best = (int)i;
        if (best < 0)
          for (size_t i = 0; i < nt; ++i)
            if (!used[i] && (best < 0 || est(i) < est(best))) best = (int)i;
        if (!mat[best]) {
          rels[best] = materialize(ctx, best, pushdown[best], needOf(best), -1, notes);
          mat[best] = true;
        }
        Rel j;
        if (exprEquiJoin(cur, rels[best], residual, j)) cur = std::move(j);
        else cur = crossJoin(cur, rels[best], notes);
        used[best] = true;
        rels[best] = Rel();
        uniq.clear();
        continue;
      }
      if (!mat[best]) {
        std::vector<JoinFilter> jfs;
        std::vector<DevVec<uint32_t>> bitmaps;
        bitmaps.reserve(edges.size());
        for (auto& e : edges) {
          const bool aCur = cur.tabs.count(e.at) && e.bt == best, bCur = cur.tabs.count(e.bt) && e.at == best;
          if (!aCur && !bCur) continue;
          const int ct = aCur ? e.at : e.bt, cc = aCur ? e.ac : e.bc, oc = aCur ? e.bc : e.ac;
          // Only worth it when the keys feeding the filter were already reduced (by filters or
          // joins) to under half of their table; a filter built from every key removes nothing.
          if ((double)cur.n > 0.5 * (double)ctx.tabs[ct]->liveRows()) continue;
          const int ri = cur.find(ct, cc);
          int64_t lo, hi;
          if (ri < 0 || cur.cols[ri].pt == PType::F64 || ctx.tabs[best]->cols[oc].pt == PType::F64 ||
              ctx.tabs[ct]->cols[cc].lt == LType::VARCHAR ||  // dictionary codes differ per table
              !colRange(*ctx.tabs[ct], cc, lo, hi) || span64(lo, hi) >= (uint64_t)1 << 30)
            continue;
          const int64_t nbits = (int64_t)span64(lo, hi) + 1;
          bitmaps.emplace_back((nbits + 31) / 32);
          gpuKeyBitmap(viewOf(cur.cols[ri].data.p, cur.cols[ri].pt), cur.n, lo, nbits, bitmaps.back().p);
          jfs.push_back({oc, lo, nbits, bitmaps.back().p,
                         ctx.tabs[best]->cols[oc].name + " IN (" + std::to_string(cur.n) + " keys of " +
                             ctx.tabs[ct]->cols[cc].name + ", " + std::to_string((nbits + 7) / 8 / 1024) + " KiB bitmap)"});
        }
        rels[best] = materialize(ctx, best, pushdown[best], needOf(best), -1, notes, jfs);
        mat[best] = true;
      }
      auto& e = edges[bestEdge];
      edgeUsed[bestEdge] = true;
      bool curHasA = cur.tabs.count(e.at) > 0;
      {
        std::pair<int, int> curCol = curHasA ? std::make_pair(e.at, e.ac) : std::make_pair(e.bt, e.bc);
        int otherCol = curHasA ? e.bc : e.ac;
        bool newUnique = otherCol == ctx.tabs[best]->pk, curUnique = uniq.count(curCol) > 0;
        std::set<std::pair<int, int>> nu;
        if (newUnique) nu = uniq;
        if (curUnique && ctx.tabs[best]->pk >= 0) nu.insert({best, ctx.tabs[best]->pk});
        uniq.swap(nu);
      }
      const Dict* da = ctx.tabs[e.at]->cols[e.ac].lt == LType::VARCHAR ? ctx.tabs[e.at]->dicts[e.ac].get() : nullptr;
      const Dict* db = ctx.tabs[e.bt]->cols[e.bc].lt == LType::VARCHAR ? ctx.tabs[e.bt]->dicts[e.bc].get() : nullptr;
      cur = curHasA ? hashJoin(cur, rels[best], e.at, e.ac, e.bt, e.bc, notes, da, db)
                    : hashJoin(cur, rels[best], e.bt, e.bc, e.at, e.ac, notes, db, da);
      used[best] = true;
      rels[best] = Rel();
    }
    if (prefixN < nt) return buildOrdered(std::move(cur), prefixN, edgeUsed);
    for (size_t ei = 0; ei < edges.size(); ++ei)
      if (!edgeUsed[ei]) residual.push_back(edges[ei].e);
    return cur;
  };

  // ================= Aggregation queries
  if (aggQuery) {
    for (auto& it : items) collectAggs(it.e);
    if (having) collectAggs(having);
    for (auto& o : orderBy) collectAggs(o.e);
    for (auto& g : groupBy)
      if (hasAgg(g)) throw SqlError("Can't group on aggregate", 1056);
    plan.nkeys = (int)groupBy.size();
    for (auto& g : groupBy)
      if (g->ty == LType::VARCHAR && !ctx.dict(*g))
        throw SqlError("GROUP BY on this string expression is not supported (it combines several columns)", 1235);
    if (distX) {  // only COUNT(DISTINCT x) aggregates: each is the count of distinct x per group (slot 0)
      if (distX->ty == LType::VARCHAR && !ctx.dict(*distX))
        throw SqlError("GROUP BY on this string expression is not supported (it combines several columns)", 1235);
      if (ctx.tabs.size() != 1) throw SqlError("internal: COUNT(DISTINCT) GPU path needs one table", 1105);
      if (ciColl() && distX->ty == LType::VARCHAR) {  // case variants are one value
        Dict* d = ctx.dict(*distX);
        if (auto cmap = d ? ciCanon(*d) : nullptr) {
          ExprP m = boundNode(Expr::FUNC, "#MAP", {distX}, LType::VARCHAR);
          m->aux = cmap;
          ctx.exprDict[m.get()] = d;
          distX = m;
        }
      }
      // COUNT(DISTINCT x) reads the extra slot phase 2 adds after the ordinary aggregates.
      for (auto& [k, r] : aggs)
        if (r.distinct) {
          r.fn = "COUNT";
          r.slot = plan.naggs;
          r.cntSlot = 0;
        }
    }

    bool single = ctx.tabs.size() == 1;
    Rel joined;
    if (!single) {
      notes.push_back("GPU join pipeline (EXPLAIN runs the joins to report the actual strategy):");
      joined = buildJoin();
      if (explain) return explainResult();
    }
    ExprP filter = single ? where : andAll(residual);
    Compiler comp(&ctx);
    plan.hasWhere = (bool)filter;
    if (filter) plan.where = comp.compile(filter);
    for (int k = 0; k < plan.nkeys; ++k) {
      // DECIMAL arithmetic runs in doubles: round the key to its scale so equal decimals group together
      // (-30.76 can come out as -30.759999.. for one row and -30.760000..1 for another).
      ExprP kx = groupBy[k];
      if (kx->ty == LType::DECIMAL && kx->kind != Expr::COL && kx->scale >= 0) {
        kx = mkCall(FN_ROUND2, {kx, boundLit(Value::I(kx->scale))}, LType::DECIMAL);
        kx->scale = groupBy[k]->scale;
      }
      plan.keys[k] = comp.compile(kx);
      plan.keyInt[k] = groupBy[k]->ty != LType::DOUBLE && groupBy[k]->ty != LType::DECIMAL && progIsInt(comp.env, plan.keys[k]);
    }
    for (int a = 1; a < plan.naggs; ++a) plan.aggs[a] = comp.compile(aggArgs[a]);
    Prog distProg;
    bool distInt = false;
    if (distX) {
      distProg = comp.compile(distX);
      distInt = distX->ty != LType::DOUBLE && distX->ty != LType::DECIMAL && progIsInt(comp.env, distProg);
    }
    comp.uploadSets();
    plan.aggInt[0] = 1;
    for (int a = 1; a < plan.naggs; ++a) {
      plan.aggInt[a] = plan.ops[a] != AGG_COUNT && progIsInt(comp.env, plan.aggs[a]);
      plan.aggNull[a] = 0;
      if (plan.ops[a] != AGG_COUNT && plan.aggs[a].len == 1 && comp.env.code[plan.aggs[a].start].op == OP_COL) {
        const uint8_t t = comp.env.cols[comp.env.code[plan.aggs[a].start].arg].t;
        if (t == 3 || t == 4) plan.aggNull[a] = t, plan.aggInt[a] = 1;  // nullable int column: exact, NULLs skipped
      }
      // GROUP BY over 64-bit integer columns: exact int64 results as bits (a SUM as its two halves,
      // see aggPart). Without GROUP BY integer sums are 128-bit and exact anyway.
      plan.aggBits[a] = 0;
      if ((plan.nkeys || distX) && plan.aggInt[a] && aggArgs[a] && aggArgs[a]->kind == Expr::COL &&
          (aggArgs[a]->ty == LType::BIGINT || aggArgs[a]->ty == LType::DATETIME) &&
          (plan.ops[a] == AGG_MIN || plan.ops[a] == AGG_MAX || plan.aggPart[a]))
        plan.aggBits[a] = 1;
    }
    for (int a = 1; a < plan.naggs; ++a)  // both halves exact, or neither (then the low slot sums whole values)
      if (plan.aggPart[a] == 1 && plan.aggBits[a] != plan.aggBits[a + 1]) plan.aggBits[a] = plan.aggBits[a + 1] = 0;
    bool anyBits = false;
    for (int a = 1; a < plan.naggs; ++a) anyBits |= plan.aggBits[a] != 0;
    const int ncols = (int)comp.slotCols.size();

    std::vector<double> keys, accs;
    int64_t groups = 0;
    // Size the GPU hash table from statistics: dictionary sizes and segment min/max ranges give
    // an upper bound on the number of groups; fall back to the input row count.
    int64_t rowsIn = single ? ctx.tabs[0]->liveRows() : joined.n;
    double bound = 1;
    for (auto& g : groupBy) {
      if (g->kind != Expr::COL || g->ty == LType::DOUBLE) { bound = 1e18; break; }
      Table& gt = *ctx.tabs[g->bt];
      if (g->ty == LType::VARCHAR) { bound *= std::max<size_t>(gt.dicts[g->bc]->strs.size(), 1); continue; }
      double mn = INFINITY, mx = -INFINITY;
      for (auto& sp : gt.segs) mn = std::min(mn, sp->mn[g->bc]), mx = std::max(mx, sp->mx[g->bc]);
      bound *= mx >= mn ? mx - mn + 1 : 1;
      bound += (double)gt.bufferTotal();
    }
    double est = std::min<double>(bound, (double)std::max<int64_t>(rowsIn, 1));

    // Dense ("encoded") GROUP BY when every key is an integer-like column with a small known range:
    // dictionary codes, small ints, dates. Ranges come from dictionary sizes, segment min/max and
    // the write buffers.
    DenseSpec dspec;
    if (plan.nkeys && tDense) {
      int64_t G = 1;
      bool ok = true;
      for (int k = 0; k < plan.nkeys && ok; ++k) {
        const ExprP& g = groupBy[k];
        if (g->kind != Expr::COL || g->ty == LType::DOUBLE) { ok = false; break; }
        int64_t lo, hi;
        if (ctx.tabs[g->bt]->hasNull[g->bc] || ctx.outerTabs.count(g->bt) || !colRange(*ctx.tabs[g->bt], g->bc, lo, hi)) {
          ok = false;
          break;
        }
        if (span64(lo, hi) >= (uint64_t)1 << 26) { ok = false; break; }
        const int64_t range = (int64_t)span64(lo, hi) + 1;
        if (G > (int64_t)(1ll << 26) / range) { ok = false; break; }
        dspec.kmin[k] = lo;
        dspec.range[k] = range;
        dspec.stride[k] = G;
        G *= range;
      }
      if (ok) {
        int off = 0;
        for (int a = 0; a < plan.naggs; ++a) {
          const uint8_t op = plan.ops[a];
          const bool in = plan.aggInt[a];
          dspec.kind[a] = op == AGG_COUNT ? DK_COUNT
                        : op == AGG_SUM   ? (in ? DK_SUM_I : DK_SUM_D)
                        : op == AGG_MIN   ? (in ? DK_MIN_I : DK_MIN_D)
                                          : (in ? DK_MAX_I : DK_MAX_D);
          dspec.slotOff[a] = off;
          int64_t bytes = G * (dspec.kind[a] == DK_COUNT ? 4 : 8);
          off += (int)std::min<int64_t>((bytes + 7) / 8 * 8, 1 << 30);
        }
        const int64_t copyBytes = off;
        // Shared-memory atomics on a handful of hot groups serialize, and FP64 atomics are
        // compare-and-swap loops on consumer GPUs: with few groups and floating-point sums the
        // warp-sort hash path (no atomics per row) is faster (Q1: 13 ms vs 42 ms).
        bool intOnly = true;
        for (int a = 0; a < plan.naggs; ++a)
          intOnly &= dspec.kind[a] == DK_COUNT || dspec.kind[a] == DK_SUM_I || dspec.kind[a] == DK_MIN_I ||
                     dspec.kind[a] == DK_MAX_I;
        if (G * plan.naggs <= 88) {  // 88 x 8 B x 128 threads = 88 KB, leaving room for static shared memory
          // Few groups: per-thread private accumulators (no atomics per row, see aggDensePrivBody).
          dspec.priv = 1;
          dspec.G = G;
        } else if (copyBytes <= 48 * 1024 && G < 64 && !intOnly) {
          // leave dspec.G = 0: hash aggregation
        } else if (copyBytes <= 48 * 1024) {
          dspec.copies = (int)std::min<int64_t>(8, (48 * 1024) / copyBytes);
          dspec.copyBytes = (int)copyBytes;
          dspec.G = G;
        } else if (G <= std::max<int64_t>(4 * rowsIn, 1 << 16) &&
                   G * (8 + 8 * plan.nkeys + 8 * plan.naggs) <= (int64_t)512 << 20) {
          dspec.copies = 0;  // straight to global memory (still no hashing)
          dspec.G = G;
        }
      }
    }
    int64_t cap = 1;
    if (plan.nkeys) {
      cap = 1024;
      int64_t maxCap = (1ll << 30) / (8 + 8 * plan.nkeys + 8 * plan.naggs);  // keep the table <= 1 GiB
      while (cap < est * 2 && cap * 2 <= maxCap) cap <<= 1;
    }
    // HAVING / ORDER BY .. LIMIT over many groups: evaluate on the GPU over the compacted groups
    // and download only the surviving (top-N) groups. The host code below then re-checks and
    // orders that small set exactly. Returns false (host fallback) for anything it can't compile.
    auto gpuPostAggregate = [&](GpuAggregator& agg, std::vector<double>& hk, std::vector<double>& ha, int64_t& out) -> bool {
      std::vector<std::string> keyCanon;
      for (auto& g : groupBy) keyCanon.push_back(canon(g));
      Compiler pc(&ctx);
      const int nk = plan.nkeys;
      pc.hook = [&](const ExprP& e) {
        std::string c = canon(e);
        for (int k = 0; k < nk; ++k)
          if (keyCanon[k] == c) { pc.emit(OP_COL, k, 1); return true; }
        if (e->kind == Expr::FUNC && isAggName(e->op)) {
          const AggRef& r = aggs.at(c);
          if (r.fn == "COUNT") pc.emit(OP_COL, nk + r.slot, 1);
          else if (r.fn == "AVG") { pc.emit(OP_COL, nk + r.slot, 1); pc.emit(OP_COL, nk + r.cntSlot, 1); pc.emit(OP_DIV, 0, -1); }
          else if (r.argTy == LType::VARCHAR && !r.dict) throw SqlError("string aggregate", 1235);
          else {
            pc.emit(OP_COL, nk + r.slot, 1);
            if (r.decScale >= 0) {
              pc.emit(OP_CONST, pc.constant(pow10i(r.decScale)), 1);
              pc.emit(OP_DIV, 0, -1);
            }
            // No non-NULL value in the group: NULL (the slot holds the accumulator's start value):
            // + 0 / count, NaN when the count is 0.
            pc.emit(OP_CONST, pc.constant(0.0), 1);
            pc.emit(OP_COL, nk + r.cntSlot, 1);
            pc.emit(OP_DIV, 0, -1);
            pc.emit(OP_ADD, 0, -1);
          }
          return true;
        }
        if (e->kind == Expr::COL) throw SqlError("not a group key", 1055);
        return false;
      };
      Prog hp;
      std::vector<Prog> ops;
      bool topN = !orderBy.empty() && s.limit >= 0;
      const bool firstN = orderBy.empty() && s.limit >= 0 && !s.distinct;
      if (!having && !topN && !firstN) return false;
      try {
        if (having) hp = pc.compile(having);
        if (topN)
          for (auto& o : orderBy) {
            if (o.e->ty == LType::VARCHAR && !ctx.dict(*o.e)) return false;
            ops.push_back(pc.compile(o.e));
          }
      } catch (const SqlError&) {
        return false;
      }
      DevVec<double> dk, da;
      const int64_t g = agg.compact(dk, da);
      std::vector<DevVec<double>> keyConv;  // integer keys (stored as int64 bits) as doubles for evaluation
      for (int k = 0; k < nk; ++k) {
        if (plan.keyInt[k]) {
          keyConv.emplace_back(std::max<int64_t>(g, 1));
          gpuBitsToDouble(dk.p + (int64_t)k * g, g, keyConv.back().p);
          pc.env.cols[k] = viewOf(keyConv.back().p, PType::F64);
        } else {
          pc.env.cols[k] = viewOf(dk.p + (int64_t)k * g, PType::F64);
        }
      }
      // Exact int64 slots (AggPlan::aggBits: SUM / MIN / MAX of integers) hold int64 bit patterns:
      // as doubles for the evaluation, like the integer keys.
      const auto bitsSlots = agg.sumBitsSlots();
      std::vector<DevVec<double>> accConv;
      accConv.reserve(plan.naggs);
      for (int a = 0; a < plan.naggs; ++a) {
        if (a < (int)bitsSlots.size() && bitsSlots[a]) {
          accConv.emplace_back(std::max<int64_t>(g, 1));
          gpuBitsToDouble(da.p + (int64_t)a * g, g, accConv.back().p);
          pc.env.cols[nk + a] = viewOf(accConv.back().p, PType::F64);
        } else {
          pc.env.cols[nk + a] = viewOf(da.p + (int64_t)a * g, PType::F64);
        }
      }
      pc.uploadSets();
      DevVec<int32_t> sel;
      int64_t m = g;
      if (having) {
        sel = DevVec<int32_t>(std::max<int64_t>(g, 1));
        m = gpuSelect(pc.env, true, hp, g, nullptr, sel.p);
      }
      DevVec<int32_t> cand;
      int64_t mc = m;
      if (topN && m > 0) {
        mc = std::min<int64_t>(m, s.offset + s.limit);
        std::vector<DevVec<double>> bufs;
        std::vector<std::shared_ptr<DevVec<int32_t>>> ranks;
        std::vector<std::pair<const double*, bool>> sk;
        for (size_t k = 0; k < orderBy.size(); ++k) {
          bufs.emplace_back(m);
          gpuEvalAt(pc.env, ops[k], having ? sel.p : nullptr, m, bufs.back().p);
          const ExprP& oe = orderBy[k].e;
          bool rankAlready = false;  // string MIN / MAX: the slot holds a rank already (not a code)
          if (oe->kind == Expr::FUNC && isAggName(oe->op))
            if (auto it = aggs.find(canon(oe)); it != aggs.end() && it->second.unrank) rankAlready = true;
          if (oe->ty == LType::VARCHAR && !rankAlready) {
            ranks.push_back(dictRankDev(*ctx.dict(*orderBy[k].e)));
            gpuMapRank(bufs.back().p, ranks.back()->p, m);
          }
          sk.push_back({bufs.back().p, orderBy[k].desc});
        }
        DevVec<int32_t> perm = gpuSortPerm(sk, m);
        cand = DevVec<int32_t>(std::max<int64_t>(mc, 1));
        if (having) gpuGather(sel.p, PType::I32, perm.p, mc, cand.p);
        else d2d(cand.p, perm.p, mc * 4);
      } else if (firstN) {  // LIMIT without ORDER BY: any offset + limit groups do
        mc = std::min<int64_t>(m, s.offset + s.limit);
        if (having) {
          cand = std::move(sel);
        } else {
          std::vector<int32_t> first((size_t)std::max<int64_t>(mc, 1));
          for (int64_t i = 0; i < mc; ++i) first[i] = (int32_t)i;
          cand = DevVec<int32_t>((int64_t)first.size());
          h2d(cand.p, first.data(), first.size() * 4);
        }
      } else {
        cand = std::move(sel);
      }
      // Download the candidate groups in one transfer, then lay them out row-major like fetch().
      const int W = nk + plan.naggs;
      DevVec<double> tmp(std::max<int64_t>(mc * W, 1));
      for (int c = 0; c < W; ++c)
        gpuGather((c < nk ? dk.p + (int64_t)c * g : da.p + (int64_t)(c - nk) * g), PType::F64, cand.p, mc, tmp.p + (int64_t)c * mc);
      const std::vector<double> all = tmp.download(mc * W);
      hk.assign((size_t)mc * nk, 0.0);
      ha.assign((size_t)mc * plan.naggs, 0.0);
      for (int c = 0; c < W; ++c)
        for (int64_t i = 0; i < mc; ++i) {
          const double v = all[(size_t)c * mc + i];
          if (c < nk) hk[(size_t)i * nk + c] = v;
          else ha[(size_t)i * plan.naggs + (c - nk)] = v;
        }
      out = mc;
      notes.push_back("  GPU post-aggregation: " + std::to_string(g) + " groups -> " + std::to_string(mc) + " sent to host");
      return true;
    };
    const std::string method =
        !plan.nkeys ? "no GROUP BY"
        : dspec.G   ? "dense GROUP BY on encoded keys (" + std::to_string(dspec.G) + " slots, " +
                        (dspec.priv ? std::string("per-thread accumulators") : dspec.copies ? std::to_string(dspec.copies) + " shared-memory copies per block" : "global arrays") + ")"
                    : "hash GROUP BY";
    const JitProgram* jit = nullptr;
    std::string codegen = jitEnabled() ? "compiled" : "interpreted (SET warpstore_codegen = OFF)";
    if (!explain && !distX) {
      ProfStage pcg("codegen");
      const JitFn fn = !plan.nkeys ? JF_NOKEYS : dspec.G ? JF_DENSE : JF_GROUPED;
      plan.densePriv = dspec.G ? (dspec.priv ? 1 : 2) : 0;
      // Plans beyond the interpreter's limits must be compiled before they can run.
      const bool mustCompile = plan.naggs > kInterpAggs || plan.nkeys > kInterpKeys;
      const JitMode savedMode = jitMode();
      if (mustCompile && savedMode != JIT_SYNC) jitSetMode(JIT_SYNC);
      jit = jitGet(comp.env, ncols, plan, fn,
                   !plan.nkeys ? "aggregate (no group by)" : dspec.G ? "aggregate (dense group by)" : "aggregate (hash group by)");
      if (mustCompile && savedMode != JIT_SYNC) jitSetMode(savedMode);
      if (jit) codegen = "compiled plan #" + std::to_string(jit->id);
      else if (jitEnabled()) codegen = "interpreted while the plan compiles in the background";
      pcg.done(-1, codegen);
    }
    ScanStats st;
    // COUNT(DISTINCT x): phase 1 groups by (keys, x) on the GPU, phase 2 aggregates the compacted
    // distinct pairs by the keys (COUNT, skipping NULL x), still on the GPU.
    std::unique_ptr<GpuAggregator> agg2;
    std::vector<uint8_t> sumBits;  // slots holding exact sums as bits (no GROUP BY)
    std::vector<int64_t> sumHi;    // their high 64 bits
    DevVec<double> distKeys, distAccs;
    if (distX && !explain) {
      AggPlan plan1 = plan;  // every ordinary aggregate as a partial per (keys, x)
      plan1.nkeys = plan.nkeys + 1;
      plan1.keys[plan.nkeys] = distProg;
      plan1.keyInt[plan.nkeys] = distInt;
      plan1.densePriv = 0;
      if (plan.naggs + 1 > kInterpAggs) throw SqlError("too many aggregates next to COUNT(DISTINCT)", 1235);
      if (plan1.nkeys > kMaxKeys) throw SqlError("too many GROUP BY expressions", 1235);
      const JitProgram* jit1 = nullptr;
      {
        ProfStage pcg("codegen");
        jit1 = jitGet(comp.env, ncols, plan1, JF_GROUPED, "aggregate (distinct pairs)");
        pcg.done(-1, jit1 ? "compiled plan #" + std::to_string(jit1->id) : "interpreted");
      }
      int64_t cap1 = 1024;
      while (cap1 < std::min<int64_t>(std::max<int64_t>(rowsIn, 1), 1 << 22)) cap1 <<= 1;
      for (;;) {
        GpuAggregator a1(plan1, cap1, nullptr);
        ProfStage p1("scan+distinct pairs " + ctx.tabs[0]->name);
        st = ScanStats();
        forEachBatch(ctx, 0, comp, conj, st, [&](ExprEnv& env, int64_t n, const uint32_t* del) {
          a1.addBatch(env, ncols, n, del);
          if (scanWindowFull()) scanFlush(), a1.runWindow(comp.env, jit1), releaseScanTemps();  // (disk tier)
          return true;
        });
        a1.run(comp.env, jit1);
        if (a1.overflowed()) {
          int64_t want = std::max<int64_t>(cap1 * 4, a1.groupCount() * 4);
          cap1 = 1;
          while (cap1 < want) cap1 <<= 1;
          continue;
        }
        const int64_t g1 = a1.compact(distKeys, distAccs);
        p1.done(a1.rowsIn(), std::to_string(g1) + " distinct (group, value) pairs, hash GROUP BY");
        ProfStage p2("count distinct");
        const int nk = plan.nkeys;
        ExprEnv env2{};
        int nc = 0;
        for (int k = 0; k <= nk; ++k) {
          const bool in = k < nk ? plan.keyInt[k] != 0 : distInt;
          env2.cols[k] = ColView{distKeys.p + (int64_t)k * g1, (uint8_t)(in ? 1 : 2)};
          env2.code[nc++] = Instr{OP_COL, (uint8_t)in, (uint16_t)k};
        }
        AggPlan plan2{};
        plan2.nkeys = nk;
        for (int k = 0; k < nk; ++k) {
          plan2.keys[k] = Prog{(int16_t)k, 1};
          plan2.keyInt[k] = plan.keyInt[k];
        }
        // Combine the partials: COUNT and SUM by SUM, MIN by MIN, MAX by MAX; the extra slot counts
        // the pairs whose x is not NULL (= COUNT(DISTINCT x) per group).
        const int na = plan1.naggs;
        for (int a = 0; a < na; ++a) {
          // (exact int64 partials, AggPlan::aggBits, are read as int64 and merged exactly)
          env2.cols[nk + 1 + a] = ColView{distAccs.p + (int64_t)a * g1, (uint8_t)(plan1.aggBits[a] ? 1 : 2)};
          plan2.aggs[a] = Prog{(int16_t)nc, 1};
          env2.code[nc++] = Instr{OP_COL, 0, (uint16_t)(nk + 1 + a)};
          plan2.ops[a] = plan1.ops[a] == AGG_COUNT ? AGG_SUM : plan1.ops[a];
          plan2.aggInt[a] = plan2.aggBits[a] = plan1.aggBits[a];
          plan2.aggPart[a] = 0;
        }
        plan2.aggs[na] = Prog{(int16_t)nc, distInt ? (int16_t)1 : (int16_t)3};
        if (distInt) {
          env2.code[nc++] = Instr{OP_CONST, 0, 0};
          env2.consts[0] = 1.0;
        } else {
          env2.code[nc++] = Instr{OP_COL, 0, (uint16_t)nk};
          env2.code[nc++] = Instr{OP_ISNULL, 0, 0};
          env2.code[nc++] = Instr{OP_NOT, 0, 0};
        }
        plan2.ops[na] = AGG_SUM;
        plan2.aggInt[na] = 0;
        plan2.naggs = na + 1;
        int64_t cap2 = 1024;
        while (cap2 < 2 * g1) cap2 <<= 1;
        agg2 = std::make_unique<GpuAggregator>(plan2, cap2, nullptr);
        agg2->addBatch(env2, nk + 1 + na, g1, nullptr);
        agg2->run(env2, nullptr);
        agg2->overflowed();
        p2.done(g1, "second GPU aggregation over the distinct pairs");
        break;
      }
      plan.naggs += 1;  // the result layout is plan2's: ordinary slots + the distinct count
      ProfStage ppost("post-aggregate");
      if (!(plan.nkeys && (having || s.limit >= 0) && gpuPostAggregate(*agg2, keys, accs, groups)))
        groups = agg2->fetch(keys, accs);
      sumBits = agg2->sumBitsSlots();
      sumHi = agg2->sumHigh();
      ppost.done(groups, plan.nkeys ? "groups sent to host" : "");
    }
    for (;;) {
      if (distX) break;
      GpuAggregator agg(plan, cap, dspec.G ? &dspec : nullptr);
      ProfStage pscan(single ? "scan+aggregate " + ctx.tabs[0]->name : "aggregate join result");
      if (single) {
        st = ScanStats();
        if (explain) {
          forEachBatch(ctx, 0, comp, conj, st, nullptr, true);
        } else {
          forEachBatch(ctx, 0, comp, conj, st, [&](ExprEnv& env, int64_t n, const uint32_t* del) {
            agg.addBatch(env, ncols, n, del);
            if (scanWindowFull()) scanFlush(), agg.runWindow(comp.env, jit), releaseScanTemps();  // (disk tier)
            return true;
          });
        }
      } else {
        bindRelCols(comp, joined);
        agg.addBatch(comp.env, ncols, joined.n, nullptr);
      }
      if (explain) break;
      agg.run(comp.env, jit);
      const bool overflow = agg.overflowed();
      pscan.done(agg.rowsIn(), std::to_string(agg.batches()) + " batches in " + std::to_string(agg.windows() + 1) + (agg.windows() ? " launches (disk tier windows), " : " launch, ") + method + ", " + codegen +
                                   (overflow ? ", hash table overflowed: retrying bigger" : ""));
      if (overflow) {
        int64_t want = std::max<int64_t>(cap * 4, agg.groupCount() * 4);
        cap = 1;
        while (cap < want) cap <<= 1;
        continue;
      }
      ProfStage ppost("post-aggregate");
      // A derived table's GROUP BY whose items are keys and plain aggregates: built on the GPU from the
      // compacted groups, never leaving VRAM (makeTemp takes the columns over).
      if (tColumnar && plan.nkeys && !having && orderBy.empty() && s.limit < 0 && s.offset == 0 && !explain) {
        std::vector<TempColSpec> specs;
        std::vector<std::shared_ptr<Dict>> dicts(items.size());
        bool ok = true;
        for (size_t i = 0; i < items.size() && ok; ++i) {
          const ExprP& e = items[i].e;
          TempColSpec sp;
          const LType ty = e->ty == LType::DECIMAL ? LType::DOUBLE : e->ty;
          if (ty == LType::VECTOR) { ok = false; break; }
          sp.out = ty == LType::DOUBLE ? 2 : physOf(ty) == PType::I64 ? 1 : 0;
          const std::string k = canon(e);
          int key = -1;
          for (int g = 0; g < plan.nkeys; ++g)
            if (canon(groupBy[g]) == k) key = g;
          if (key >= 0) {
            if (ty == LType::VARCHAR) {  // the derived table shares the dictionary (it only grows), or a copy
              const Dict* d = ctx.dict(*e);
              if (!d) { ok = false; break; }
              if (e->kind == Expr::COL && e->bt >= 0) dicts[i] = ctx.tabs[e->bt]->dicts[e->bc];
              for (auto& sd : ctx.sharedDicts)
                if (!dicts[i] && sd.get() == d) dicts[i] = sd;
              for (auto& tb : ctx.tabs)  // (e.g. canonical case-insensitive codes of a column: its dictionary)
                for (auto& td : tb->dicts)
                  if (!dicts[i] && td.get() == d) dicts[i] = td;
              if (!dicts[i]) {
                if (d->strs.size() > (1u << 20)) { ok = false; break; }
                dicts[i] = std::make_shared<Dict>(*d);
              }
            }
            sp.src = 0;
            sp.idx = key;
            sp.keyInt = plan.keyInt[key];
          } else if ((e->kind == Expr::FUNC && isAggName(e->op) && ty != LType::VARCHAR) ||
                     (e->kind == Expr::BIN && e->op == "*" && ty == LType::DOUBLE &&
                      ((e->args[0]->kind == Expr::LIT && e->args[1]->kind == Expr::FUNC && isAggName(e->args[1]->op)) ||
                       (e->args[1]->kind == Expr::LIT && e->args[0]->kind == Expr::FUNC && isAggName(e->args[0]->op))))) {
            ExprP ag = e;
            if (e->kind == Expr::BIN) {  // constant * aggregate
              const bool litFirst = e->args[0]->kind == Expr::LIT;
              const Value& lv = e->args[litFirst ? 0 : 1]->lit;
              if (lv.isNull() || lv.kind == Value::STR) { ok = false; break; }
              sp.mul = lv.num();
              ag = e->args[litFirst ? 1 : 0];
            }
            auto it = aggs.find(canon(ag));
            if (it == aggs.end() || it->second.unrank || it->second.distinct || it->second.dict) { ok = false; break; }
            const AggRef& r = it->second;
            const bool bits = plan.aggBits[r.slot];
            sp.src = 1;
            sp.idx = r.slot;
            sp.cnt = r.fn == "COUNT" ? -1 : r.cntSlot;
            sp.div = r.decScale >= 0 ? pow10i(r.decScale) : 1.0;
            if (r.fn == "COUNT") sp.kind = 3;
            else if (r.fn == "AVG") sp.kind = 4, sp.sumBits = bits, sp.hi = bits ? r.hiSlot : -1;
            else if (bits) sp.kind = r.hiSlot >= 0 ? 2 : 1, sp.hi = r.hiSlot;
            else sp.kind = 0;
          } else {
            ok = false;
          }
          specs.push_back(sp);
        }
        if (ok) {
          DevVec<double> dk, da;
          const int64_t g = agg.compact(dk, da);
          for (auto& sp : specs) {
            auto col = std::make_shared<ResultSet::DevColumn>();
            col->p = gpuTempColumn(dk.p, plan.nkeys, da.p, plan.naggs, g, sp, col->mn, col->mx);
            res.devCols.push_back(col);
          }
          res.devRows = g;
          res.colDicts = dicts;
          for (auto& t : res.types)
            if (t == LType::DECIMAL) t = LType::DOUBLE;
          res.isResult = true;
          res.typed = true;
          ppost.done(g, "derived table built on the GPU");
          res.info = single ? "gpu-aggregate" : "gpu-join-aggregate";
          return res;
        }
      }
      if (!(plan.nkeys && !anyBits && (having || s.limit >= 0) &&
            gpuPostAggregate(agg, keys, accs, groups)))
        groups = agg.fetch(keys, accs);
      sumBits = agg.sumBitsSlots();
      sumHi = agg.sumHigh();
      ppost.done(groups, plan.nkeys ? "groups sent to host" : "");
      break;
    }
    notes.insert(notes.begin(), std::string(single ? "GPU fused scan+filter+aggregate" : "GPU aggregate") + ": " + method +
                                    ", " + std::to_string(plan.naggs) + " agg slots, codegen " +
                                    (jitEnabled() ? "on" : "off"));
    if (single) {
      notes.insert(notes.begin() + 1, "  " + scanDesc(*ctx.tabs[0], st));
      if (where) notes.insert(notes.begin() + 2, "  filter: " + where->str());
    }
    if (explain) return explainResult();
    ProfStage phost("host finalize");

    // Non-NULL counts (SUM(ISNOTNULL x) slots) are read as plain doubles everywhere: convert their exact bits.
    for (size_t a = 1; a < sumBits.size() && a < aggArgs.size() && a < accs.size(); ++a)
      if (sumBits[a] && aggArgs[a] && aggArgs[a]->kind == Expr::FUNC && aggArgs[a]->op == "ISNOTNULL")
        accs[a] = (double)bitsOf(accs[a]), sumBits[a] = 0;
    PostCtx pc;
    pc.bc = &ctx;
    pc.aggs = &aggs;
    if (!sumBits.empty()) pc.accBits = sumBits.data();
    if (!sumHi.empty() && !plan.nkeys) pc.accHi = sumHi.data();
    for (int k = 0; k < plan.nkeys; ++k) pc.keyInt.push_back(plan.keyInt[k]);
    for (auto& g : groupBy) {
      pc.keyCanon.push_back(canon(g));
      pc.keyTy.push_back(g->ty);
      pc.keyDict.push_back(ctx.dict(*g));
      pc.keyScale.push_back(g->scale);
    }
    // Filter groups, compute ORDER BY keys, pick the top rows, and only then evaluate the
    // select list (keeps large GROUP BY ... ORDER BY .. LIMIT cheap on the host).
    // On all cores (millions of groups when a GROUP BY feeds a derived table): each chunk with its
    // own evaluation context, results concatenated in group order.
    std::vector<int64_t> keep;
    std::vector<std::vector<Value>> sortKeys;
    // Every fetched group is kept and in order (no HAVING / ORDER BY, compacted groups): no index arrays.
    bool identity = !having && orderBy.empty() && plan.nkeys;
    for (int64_t gi = 0; identity && gi < groups; gi += std::max<int64_t>(1, groups / 64))
      identity = accs[gi * plan.naggs] != 0;
    // ORDER BY a string GROUP BY key: compare its rank in the sorted dictionary, not the strings.
    std::vector<std::shared_ptr<std::vector<double>>> rankOf(orderBy.size());
    std::vector<int> rankKey(orderBy.size(), -1);
    for (size_t k = 0; k < orderBy.size(); ++k) {
      const ExprP& oe = orderBy[k].e;
      if (oe->ty != LType::VARCHAR || !ctx.dict(*oe)) continue;
      const std::string oc = canon(oe);
      for (int g = 0; g < plan.nkeys; ++g)
        if (pc.keyCanon[g] == oc && plan.keyInt[g]) {
          rankKey[k] = g;
          rankOf[k] = sortedDict(*ctx.dict(*oe)).second;
        }
    }
    ProfStage psub("  finalize: setup");
    psub.done();
    if (!identity) {
      const size_t nchunks = std::max<size_t>(1, std::min<size_t>(64, (size_t)groups / 16384));
      std::vector<std::vector<int64_t>> keepC(nchunks);
      std::vector<std::vector<std::vector<Value>>> skC(nchunks);
      std::exception_ptr err;
      std::mutex errMu;
      parallelFor(nchunks, [&](size_t cb, size_t ce) {
        for (size_t ch = cb; ch < ce; ++ch) {
          try {
            PostCtx tpc = pc;
            const int64_t lo = (int64_t)(groups * ch / nchunks), hi = (int64_t)(groups * (ch + 1) / nchunks);
            for (int64_t gi = lo; gi < hi; ++gi) {
              tpc.keys = keys.data() + gi * plan.nkeys;
              tpc.accs = accs.data() + gi * plan.naggs;
              if (plan.nkeys && tpc.accs[0] == 0) continue;
              if (having && !isTrue(evalPost(having, tpc))) continue;
              keepC[ch].push_back(gi);
              if (!orderBy.empty()) {
                std::vector<Value> sk;
                for (size_t k = 0; k < orderBy.size(); ++k) {
                  if (rankOf[k]) {
                    const long long code = bitsOf(tpc.keys[rankKey[k]]);
                    sk.push_back(code < 0 || code >= (long long)rankOf[k]->size() ? Value::Null() : Value::D((*rankOf[k])[code]));
                    continue;
                  }
                  sk.push_back(evalPost(orderBy[k].e, tpc));
                }
                skC[ch].push_back(std::move(sk));
              }
            }
          } catch (...) {
            std::lock_guard<std::mutex> g(errMu);
            err = std::current_exception();
          }
        }
      }, 1);
      if (err) std::rethrow_exception(err);
      for (size_t ch = 0; ch < nchunks; ++ch) {
        keep.insert(keep.end(), keepC[ch].begin(), keepC[ch].end());
        for (auto& k : skC[ch]) sortKeys.push_back(std::move(k));
      }
    }
    // (identity: verify the sampled assumption on all cores, falling back to the index arrays)
    if (identity) {
      std::atomic<bool> empty{false};
      parallelFor((size_t)groups, [&](size_t b, size_t e) {
        for (size_t gi = b; gi < e && !empty; ++gi)
          if (accs[gi * plan.naggs] == 0) empty = true;
      });
      if (empty) {
        identity = false;
        for (int64_t gi = 0; gi < groups; ++gi)
          if (accs[gi * plan.naggs] != 0) keep.push_back(gi);
      }
    }
    const size_t nkeep = identity ? (size_t)groups : keep.size();
    psub.stage = "  finalize: filter + sort keys";
    psub.done(nkeep);
    // A VARCHAR GROUP BY key can be a column of dictionary codes in the columnar result (its dictionary,
    // when small, is copied into the temp table).
    auto codeKey = [&](const ExprP& e) {
      if (e->ty != LType::VARCHAR) return false;
      const Dict* d = ctx.dict(*e);
      if (!d || d->strs.size() > (1u << 20)) return false;
      const std::string k = canon(e);
      for (auto& kc : pc.keyCanon)
        if (kc == k) return true;
      return false;
    };
    bool columnarOk = tColumnar && s.limit < 0 && s.offset == 0;  // (the condition of the columnar result below)
    for (auto& it : items) columnarOk = columnarOk && (it.e->ty != LType::VARCHAR || codeKey(it.e)) && it.e->ty != LType::VECTOR;
    if (identity && !columnarOk) {  // the general paths below index through keep
      keep.resize(nkeep);
      for (size_t i = 0; i < nkeep; ++i) keep[i] = (int64_t)i;
      identity = false;
    }
    std::vector<size_t> order(identity ? 0 : keep.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    size_t want = s.limit < 0 ? nkeep : std::min<size_t>(nkeep, (size_t)(s.offset + s.limit));
    if (!orderBy.empty()) {
      // All-numeric keys (numbers, dates, string ranks): compared as one flat array of doubles.
      // String keys: their distinct values sorted once (collation order), compared by rank.
      const size_t K = orderBy.size();
      bool flat = true;
      std::vector<std::unordered_map<std::string, double>> strRank(K);
      for (size_t k = 0; k < K && flat; ++k) {
        bool str = false, num = false;
        for (auto& sk : sortKeys) {
          const Value& v = sk[k];
          if (v.isNull()) continue;
          if (v.kind == Value::STR) str = true;
          else num = true;
          if (v.kind != Value::DBL && v.kind != Value::STR && std::llabs(v.i) > (1ll << 53)) flat = false;
        }
        if (str && num) flat = false;
        if (!str || !flat) continue;
        std::vector<const Value*> u;
        for (auto& sk : sortKeys)
          if (!sk[k].isNull() && strRank[k].emplace(sk[k].s, 0).second) u.push_back(&sk[k]);
        std::sort(u.begin(), u.end(), [](const Value* a, const Value* b) { return cmpVal(*a, *b) < 0; });
        double r = 0;
        for (size_t i = 0; i < u.size(); ++i) {
          if (i && cmpVal(*u[i - 1], *u[i]) != 0) ++r;  // (equal under the collation: same rank)
          strRank[k][u[i]->s] = r;
        }
      }
      std::vector<double> fk;
      if (flat) {
        fk.resize(sortKeys.size() * K);
        for (size_t i = 0; i < sortKeys.size(); ++i)
          for (size_t k = 0; k < K; ++k) {
            const Value& v = sortKeys[i][k];
            fk[i * K + k] = v.isNull() ? NAN : v.kind == Value::STR ? strRank[k][v.s] : v.num();
          }
      }
      auto less = [&](size_t a, size_t b) {
        if (flat) {
          const double* x = &fk[a * K];
          const double* y = &fk[b * K];
          for (size_t k = 0; k < K; ++k) {
            const bool nx = x[k] != x[k], ny = y[k] != y[k];
            const int c = nx || ny ? (nx == ny ? 0 : nx ? -1 : 1) : (x[k] < y[k] ? -1 : x[k] > y[k] ? 1 : 0);  // (NULLs first)
            if (c) return orderBy[k].desc ? c > 0 : c < 0;
          }
          return a < b;
        }
        for (size_t k = 0; k < K; ++k) {
          int c = cmpVal(sortKeys[a][k], sortKeys[b][k]);
          if (c) return orderBy[k].desc ? c > 0 : c < 0;
        }
        return a < b;
      };
      if (want * 4 >= order.size()) {
        std::sort(order.begin(), order.end(), less);
      } else {
        std::partial_sort(order.begin(), order.begin() + want, order.end(), less);
      }
    }
    psub.stage = "  finalize: sort";
    psub.done();
    std::vector<std::vector<Value>> rows, noKeys;
    bool allNumeric = tColumnar && orderBy.empty() && !having && s.limit < 0 && s.offset == 0;
    for (auto& it : items) allNumeric = allNumeric && (it.e->ty != LType::VARCHAR || codeKey(it.e)) && it.e->ty != LType::VECTOR;
    if (allNumeric) {  // a derived table's GROUP BY: numeric columns, no row objects
      const size_t nr = want;
      res.numCols.resize(items.size());
      parallelFor(items.size(), [&](size_t b, size_t e) {  // (zero-filling 10s of MB: one column per core)
        for (size_t i = b; i < e; ++i) res.numCols[i].resize(nr);
      }, 1);
      // Items that are just a GROUP BY key or a plain aggregate read the result arrays directly.
      std::vector<int> keyOf(items.size(), -1);
      std::vector<const AggRef*> aggOf(items.size(), nullptr);
      for (size_t i = 0; i < items.size(); ++i) {
        const std::string k = canon(items[i].e);
        for (size_t g = 0; g < pc.keyCanon.size(); ++g)
          if (pc.keyCanon[g] == k) keyOf[i] = (int)g;
        if (keyOf[i] < 0 && items[i].e->kind == Expr::FUNC && isAggName(items[i].e->op)) {
          auto it = aggs.find(k);
          if (it != aggs.end() && !it->second.unrank && !it->second.distinct) aggOf[i] = &it->second;
        }
      }
      std::exception_ptr err;
      std::mutex errMu;
      parallelFor(nr, [&](size_t b, size_t e) {
        try {
          PostCtx tpc = pc;
          for (size_t j = b; j < e; ++j) {
            const int64_t g = identity ? (int64_t)j : keep[order[j]];
            tpc.keys = keys.data() + g * plan.nkeys;
            tpc.accs = accs.data() + g * plan.naggs;
            for (size_t i = 0; i < items.size(); ++i) {
              if (keyOf[i] >= 0) {
                const double kv = tpc.keys[keyOf[i]];
                res.numCols[i][j] = (size_t)keyOf[i] < pc.keyInt.size() && pc.keyInt[keyOf[i]] ? (double)bitsOf(kv) : kv;  // (NULL keys: the sentinel round-trips)
                continue;
              }
              if (const AggRef* r = aggOf[i]) {
                if (r->fn == "COUNT") { res.numCols[i][j] = tpc.accs[r->slot]; continue; }
                const double cnt = tpc.accs[r->cntSlot];
                double v = tpc.accs[r->slot];
                if (pc.accBits && pc.accBits[r->slot]) {  // exact int64 bits (a BIGINT SUM: two 32-bit halves)
                  v = r->hiSlot >= 0 && pc.accBits[r->hiSlot] ? (double)bitsOf(tpc.accs[r->hiSlot]) * 4294967296.0 + (double)bitsOf(v)
                    : pc.accHi && (r->fn == "SUM" || r->fn == "AVG") ? (double)(((__int128)pc.accHi[r->slot] << 64) | (uint64_t)bitsOf(v))
                                                                    : (double)bitsOf(v);
                }
                if (r->decScale >= 0) v /= pow10i(r->decScale);
                res.numCols[i][j] = cnt == 0 ? nullv() : r->fn == "AVG" ? v / cnt : v;
                continue;
              }
              const Value v = evalPost(items[i].e, tpc);
              res.numCols[i][j] = v.isNull() ? nullv() : v.kind == Value::DBL ? v.d : (double)v.i;
            }
          }
        } catch (...) {
          std::lock_guard<std::mutex> g(errMu);
          err = std::current_exception();
        }
      }, 16384);
      if (err) std::rethrow_exception(err);
      res.typed = true;
      for (auto& t : res.types)  // (a DECIMAL expression's double value: a DOUBLE column of the temp table)
        if (t == LType::DECIMAL) t = LType::DOUBLE;
      res.colDicts.assign(items.size(), nullptr);
      for (size_t i = 0; i < items.size(); ++i)
        if (items[i].e->ty == LType::VARCHAR) {  // (shared when it is a column's or a cached derived one)
          const ExprP& e = items[i].e;
          const Dict* d = ctx.dict(*e);
          if (e->kind == Expr::COL && e->bt >= 0) res.colDicts[i] = ctx.tabs[e->bt]->dicts[e->bc];
          for (auto& sd : ctx.sharedDicts)
            if (!res.colDicts[i] && sd.get() == d) res.colDicts[i] = sd;
          for (auto& tb : ctx.tabs)
            for (auto& td : tb->dicts)
              if (!res.colDicts[i] && td.get() == d) res.colDicts[i] = td;
          if (!res.colDicts[i]) res.colDicts[i] = std::make_shared<Dict>(*d);
        }
      phost.done((int64_t)nr, "columnar result for a derived table");
      res.info = single ? "gpu-aggregate" : "gpu-join-aggregate";
      return res;
    }
    {
      const size_t first = std::min<size_t>(s.offset, want), nr = want - first;
      rows.resize(nr);
      std::exception_ptr err;
      std::mutex errMu;
      parallelFor(nr, [&](size_t b, size_t e) {
        try {
          PostCtx tpc = pc;
          for (size_t j = b; j < e; ++j) {
            const size_t i = first + j;
            tpc.keys = keys.data() + keep[order[i]] * plan.nkeys;
            tpc.accs = accs.data() + keep[order[i]] * plan.naggs;
            rows[j].reserve(items.size());
            for (auto& it : items) rows[j].push_back(evalPost(it.e, tpc));
          }
        } catch (...) {
          std::lock_guard<std::mutex> g(errMu);
          err = std::current_exception();
        }
      }, 16384);
      if (err) std::rethrow_exception(err);
    }
    orderBy.clear();
    s.offset = 0;
    s.limit = -1;
    sortKeys.clear();
    finish(rows, noKeys);
    phost.done((int64_t)res.rows.size(), "HAVING / ORDER BY / LIMIT on the candidate groups, formatting");
    res.info = single ? "gpu-aggregate" : "gpu-join-aggregate";
    if (envSetting("TRACE")) fprintf(stderr, "[trace] %.2f ms %s\n", msSince(t0), res.info.c_str());
    return res;
  }

  // ================= Plain projections (optionally joined / ordered / limited)
  // Predicates and expressions the GPU can't evaluate (host-only functions over several columns)
  // run on the host over the rows the GPU filters produce.
  std::vector<ExprP> hostFilt;
  for (auto& pd : pushdown) {
    std::vector<ExprP> keep;
    for (auto& p : pd) (gpuCompilable(p, ctx) ? keep : hostFilt).push_back(p);
    pd.swap(keep);
  }
  {
    std::vector<ExprP> keep;
    for (auto& p : residual) (gpuCompilable(p, ctx) ? keep : hostFilt).push_back(p);
    residual.swap(keep);
  }
  bool hostMode = !hostFilt.empty();
  for (auto& it : items) hostMode = hostMode || !gpuCompilable(it.e, ctx);
  for (auto& o : orderBy) hostMode = hostMode || !gpuCompilable(o.e, ctx);
  if (hostMode) {
    for (auto& f : hostFilt) colsOf(f, needed);
    notes.push_back("host evaluation of: " + [&] {
      std::string r;
      for (auto& f : hostFilt) r += (r.empty() ? "" : ", ") + f->str();
      for (auto& it : items) if (!gpuCompilable(it.e, ctx)) r += (r.empty() ? "" : ", ") + it.e->str();
      return r;
    }());
  }
  Rel rel;
  // Top-N over a table with columns on disk: only the filter / ORDER BY columns are scanned (with
  // each row's location); the other columns are read for the N rows that are left.
  std::set<int> deferred;
  std::vector<Loc> lateLocs;
  if (ctx.tabs.size() == 1 && !hostMode && !orderBy.empty() && s.limit >= 0 && s.offset + s.limit <= 10000 && !tRawNumber) {
    std::set<std::pair<int, int>> early;
    for (auto& o : orderBy) colsOf(o.e, early);
    for (auto& p : pushdown[0]) colsOf(p, early);
    for (auto& p : residual) colsOf(p, early);
    const Table& t0 = *ctx.tabs[0];
    bool cold = false;
    for (int c : needOf(0)) {
      if (early.count({0, c})) continue;
      deferred.insert(c);
      for (auto& sp : t0.segs) cold = cold || segCold(*sp->d, c);
    }
    if (!cold || deferred.size() == needOf(0).size()) deferred.clear();
  }
  if (ctx.tabs.size() == 1) {
    int64_t maxRows = orderBy.empty() && s.limit >= 0 && hostFilt.empty() ? s.offset + s.limit : -1;
    notes.push_back("GPU filter + gather (late materialization)" + std::string(maxRows >= 0 ? ", early stop at LIMIT" : ""));
    if (explain) return explainResult();
    std::set<int> need = needOf(0);
    for (int c : deferred) need.erase(c);
    rel = materialize(ctx, 0, pushdown[0], need, maxRows, notes, {}, deferred.empty() ? nullptr : &lateLocs);
    for (int c : deferred) rel.cols.push_back(RelCol{0, c, ctx.tabs[0]->cols[c].pt, DevVec<uint8_t>(16), ctx.tabs[0]->colBytes(c)});  // (filled for the top N below)
  } else {
    notes.push_back("GPU join pipeline (EXPLAIN runs the joins to report the actual strategy):");
    rel = buildJoin();
    if (explain) return explainResult();
  }
  Compiler comp(&ctx);
  ExprP filter = ctx.tabs.size() == 1 ? nullptr : andAll(residual);
  Prog fp;
  if (filter) fp = comp.compile(filter);
  std::vector<Prog> projs, ords;
  std::vector<std::pair<int, int>> hostCols;
  std::vector<Prog> hostProgs;
  if (hostMode) {
    std::set<std::pair<int, int>> hc;
    for (auto& it : items) colsOf(it.e, hc);
    for (auto& o : orderBy) colsOf(o.e, hc);
    for (auto& f : hostFilt) colsOf(f, hc);
    for (auto& pr : hc) {
      hostCols.push_back(pr);
      const bool vec = ctx.tabs[pr.first]->cols[pr.second].pt == PType::VEC;
      hostProgs.push_back(vec ? Prog{} : comp.compile(boundCol(ctx, pr.first, pr.second)));
    }
  } else {
    for (auto& it : items) projs.push_back(comp.compile(it.e));
    for (auto& o : orderBy) ords.push_back(comp.compile(o.e));
  }
  comp.uploadSets();
  bindRelCols(comp, rel);
  DevVec<int32_t> sel;
  int64_t m = rel.n;
  if (filter) {
    ProfStage pf("residual filter");
    sel = DevVec<int32_t>(rel.n);
    m = gpuSelect(comp.env, true, fp, rel.n, nullptr, sel.p);
    pf.done(m, filter->str());
  }
  if (hostMode) {
    if (m > 20'000'000) throw SqlError("too many rows for host evaluation (" + std::to_string(m) + "); add filters", 1105);
    ProfStage ph("host evaluation");
    std::vector<std::vector<double>> cv;
    std::vector<std::vector<uint8_t>> vecBytes(hostCols.size());
    for (size_t i = 0; i < hostProgs.size(); ++i) {
      const Table& tb = *ctx.tabs[hostCols[i].first];
      if (tb.cols[hostCols[i].second].pt == PType::VEC) {  // raw float32 vectors straight from the relation
        const int ri = rel.find(hostCols[i].first, hostCols[i].second);
        const int es = rel.cols[ri].esize;
        DevVec<uint8_t> tmp(std::max<int64_t>(m, 1) * es);
        if (filter) gpuGatherBytes(rel.cols[ri].data.p, es, sel.p, m, tmp.p, false);
        else if (m) d2d(tmp.p, rel.cols[ri].data.p, m * es);
        vecBytes[i] = tmp.download(m * es);
        cv.emplace_back();
        continue;
      }
      DevVec<double> out(std::max<int64_t>(m, 1));
      gpuEvalAt(comp.env, hostProgs[i], filter ? sel.p : nullptr, m, out.p);
      cv.push_back(out.download(m));
    }
    std::vector<Value> rowVals(hostCols.size());
    PostCtx pc;
    pc.row = &rowVals;
    pc.colSlot = [&](int bt, int bc) {
      for (size_t i = 0; i < hostCols.size(); ++i)
        if (hostCols[i].first == bt && hostCols[i].second == bc) return (int)i;
      return -1;
    };
    std::vector<std::vector<Value>> rows, keys;
    for (int64_t r = 0; r < m; ++r) {
      for (size_t i = 0; i < hostCols.size(); ++i) {
        const Table& tb = *ctx.tabs[hostCols[i].first];
        const ColumnMeta& cm = tb.cols[hostCols[i].second];
        if (cm.pt == PType::VEC) {
          const float* f = (const float*)vecBytes[i].data() + r * cm.vecDim;
          rowVals[i] = f[0] != f[0] ? Value::Null() : Value::S(formatVector(f, cm.vecDim));
          continue;
        }
        rowVals[i] = typedVal(cv[i][r], cm.lt, tb.dicts[hostCols[i].second].get(), cm.scale);
      }
      bool keep = true;
      for (auto& f : hostFilt) keep = keep && isTrue(evalPost(f, pc));
      if (!keep) continue;
      std::vector<Value> row, key;
      for (auto& it : items) row.push_back(evalPost(it.e, pc));
      for (auto& o : orderBy) key.push_back(evalPost(o.e, pc));
      rows.push_back(std::move(row));
      keys.push_back(std::move(key));
    }
    finish(rows, keys);
    ph.done((int64_t)res.rows.size(), std::to_string(m) + " rows from the GPU");
    res.info = "gpu-scan+host";
    return res;
  }
  DevVec<int32_t> finalIdx;
  int64_t start = std::min<int64_t>(s.offset, m);
  int64_t cnt = s.limit < 0 ? m - start : std::min<int64_t>(m - start, s.limit);
  if (tInsertCols ? cnt * (int64_t)projs.size() > 1'000'000'000 : cnt > 20'000'000)
    throw SqlError("result set too large (" + std::to_string(cnt) + " rows); add a LIMIT", 1105);
  ProfStage psort(orderBy.empty() ? "limit" : "GPU sort");
  if (!orderBy.empty() && m > 0) {
    std::vector<DevVec<double>> keyBufs;
    std::vector<std::shared_ptr<DevVec<int32_t>>> ranks;
    std::vector<std::pair<const double*, bool>> keys;
    for (size_t k = 0; k < orderBy.size(); ++k) {
      keyBufs.emplace_back(m);
      gpuEvalAt(comp.env, ords[k], filter ? sel.p : nullptr, m, keyBufs.back().p);
      if (orderBy[k].e->ty == LType::VARCHAR) {
        ranks.push_back(dictRankDev(*ctx.dict(*orderBy[k].e)));
        gpuMapRank(keyBufs.back().p, ranks.back()->p, m);
      }
      keys.push_back({keyBufs.back().p, orderBy[k].desc});
    }
    DevVec<int32_t> perm = gpuSortPerm(keys, m);
    finalIdx = DevVec<int32_t>(std::max<int64_t>(cnt, 1));
    if (filter) gpuGather(sel.p, PType::I32, perm.p + start, cnt, finalIdx.p);
    else d2d(finalIdx.p, perm.p + start, cnt * 4);
    notes.push_back("  GPU sort of " + std::to_string(m) + " rows");
  } else if (cnt > 0) {
    finalIdx = DevVec<int32_t>(cnt);
    if (filter) d2d(finalIdx.p, sel.p + start, cnt * 4);
    else {
      std::vector<int32_t> seq(cnt);
      for (int64_t i = 0; i < cnt; ++i) seq[i] = (int32_t)(start + i);
      h2d(finalIdx.p, seq.data(), cnt * 4);
    }
  }
  psort.done(cnt, std::to_string(m) + " rows in");
  if (!deferred.empty() && cnt > 0) {  // the top N rows, with the deferred columns read for them
    ProfStage plate("late materialization");
    const Table& t0 = *ctx.tabs[0];
    std::vector<int32_t> fi = finalIdx.download(cnt);
    Rel top;
    top.n = cnt;
    std::vector<CellRead> lateReads;  // all deferred cells on disk, read together
    std::vector<std::pair<void*, std::vector<uint8_t>>> lateUploads;
    for (auto& rc : rel.cols) {
      RelCol nc{rc.bt, rc.bc, rc.pt, DevVec<uint8_t>(cnt * rc.esize), rc.esize};
      if (!deferred.count(rc.bc)) {
        gpuGatherBytes(rc.data.p, rc.esize, finalIdx.p, cnt, nc.data.p, false);
      } else {
        std::vector<uint8_t> h((size_t)cnt * rc.esize);
        std::vector<CellRead> cold;
        for (int64_t r = 0; r < cnt; ++r) {
          const Loc l = lateLocs[fi[r]];
          if (l.seg >= 0 && segCold(*t0.segs[l.seg]->d, rc.bc)) cold.push_back({t0.segs[l.seg]->d.get(), rc.bc, l.row, h.data() + r * rc.esize});
          else cellRaw(t0, l, rc.bc, h.data() + r * rc.esize);
        }
        lateReads.insert(lateReads.end(), cold.begin(), cold.end());
        lateUploads.push_back({nc.data.p, std::move(h)});
      }
      top.cols.push_back(std::move(nc));
    }
    segRawCells(lateReads);
    for (auto& [dp, h] : lateUploads) h2d(dp, h.data(), h.size());
    rel = std::move(top);
    bindRelCols(comp, rel);
    std::vector<int32_t> seq(cnt);
    for (int64_t i = 0; i < cnt; ++i) seq[i] = (int32_t)i;
    h2d(finalIdx.p, seq.data(), cnt * 4);
    notes.push_back("  late materialization: " + std::to_string(deferred.size()) + " columns read for " + std::to_string(cnt) + " rows");
    plate.done(cnt, std::to_string(deferred.size()) + " columns read for the top rows");
  }
  ProfStage pproj("project + format");
  // IN (SELECT int_col ..): only the distinct values matter: deduplicated on the GPU through a bitmap
  // over the column's value range, so only those come back.
  if (tRawNumber && projs.size() == 1 && items[0].e->kind == Expr::COL && items[0].e->ty != LType::VARCHAR &&
      items[0].e->ty != LType::DOUBLE && items[0].e->ty != LType::DECIMAL && items[0].e->ty != LType::VECTOR && cnt > 4096) {
    int64_t lo, hi;
    const ExprP& ce = items[0].e;
    if (colRange(*ctx.tabs[ce->bt], ce->bc, lo, hi) && span64(lo, hi) < (uint64_t)1 << 31) {
      std::vector<int64_t> d = gpuDistinctInts(comp.env, projs[0], finalIdx.p, cnt, lo, (int64_t)span64(lo, hi) + 1);
      auto nc = std::make_shared<std::vector<double>>(d.begin(), d.end());
      res.numCol = nc;
      res.typed = true;
      pproj.done(cnt, std::to_string(d.size()) + " distinct values (GPU bitmap)");
      res.info = "gpu-scan";
      return res;
    }
  }
  std::vector<std::vector<double>> cols;
  std::vector<std::vector<int64_t>> icols(projs.size());  // exact 64-bit integer columns
  for (size_t i = 0; i < projs.size(); ++i) {
    const ExprP& e = items[i].e;
    if ((e->kind == Expr::COL || e->kind == Expr::BIN || e->kind == Expr::UNARY) && (e->ty == LType::BIGINT || e->ty == LType::DATETIME)) {
      DevVec<int64_t> out(std::max<int64_t>(cnt, 1));
      gpuEvalAtInt(comp.env, projs[i], finalIdx.p, cnt, out.p);
      icols[i] = out.download(std::max<int64_t>(cnt, 0));
      cols.emplace_back();
      continue;
    }
    DevVec<double> out(std::max<int64_t>(cnt, 1));
    gpuEvalAt(comp.env, projs[i], finalIdx.p, cnt, out.p);
    cols.push_back(out.download(std::max<int64_t>(cnt, 0)));
  }
  if (tRawNumber && projs.size() == 1 && items[0].e->ty != LType::VARCHAR && items[0].e->ty != LType::VECTOR) {
    auto nc = std::make_shared<std::vector<double>>();
    if (!icols[0].empty()) {
      nc->reserve(icols[0].size());
      for (int64_t v : icols[0]) nc->push_back(v == kNullI64 ? nullv() : (double)v);
    } else {
      *nc = std::move(cols[0]);
    }
    res.numCol = nc;
    res.typed = true;
    pproj.done(cnt);
    res.info = "gpu-scan";
    return res;
  }
  if (tInsertCols) {  // INSERT .. SELECT: the columns as they are (strings: codes of plain columns)
    bool ok = true;
    for (auto& it : items)
      ok = ok && it.e->ty != LType::VECTOR && (it.e->ty != LType::VARCHAR || (it.e->kind == Expr::COL && it.e->bt >= 0));
    if (ok) {
      res.insertCols = res.typed = true;
      res.colDicts.assign(items.size(), nullptr);
      res.colScales.assign(items.size(), -1);
      for (size_t i = 0; i < items.size(); ++i) {
        const ExprP& e = items[i].e;
        if (e->ty == LType::VARCHAR) res.colDicts[i] = ctx.tabs[e->bt]->dicts[e->bc];
        res.colScales[i] = e->scale;
      }
      res.numCols = std::move(cols);
      res.intCols = std::move(icols);
      pproj.done(cnt, "columns for INSERT .. SELECT");
      res.info = "gpu-scan";
      return res;
    }
  }
  auto valueAt = [&](size_t i, int64_t r) {
    if (!icols[i].empty()) {
      const int64_t v = icols[i][r];
      if (v == kNullI64) return Value::Null();
      return items[i].e->ty == LType::DATETIME ? Value::Ts(v) : Value::I(v);
    }
    return typedVal(cols[i][r], items[i].e->ty, ctx.dict(*items[i].e), items[i].e->scale);
  };
  res.typed = tWantVals;
  // Rows are built on all cores (big results: formatting dominates the GPU work).
  if (tWantVals) res.vals.resize(std::max<int64_t>(cnt, 0));
  else res.rows.resize(std::max<int64_t>(cnt, 0));
  const bool wantVals = tWantVals;
  parallelFor((size_t)std::max<int64_t>(cnt, 0), [&](size_t b, size_t e) {
    for (size_t r = b; r < e; ++r) {
      if (wantVals) {
        std::vector<Value>& row = res.vals[r];
        row.reserve(items.size());
        for (size_t i = 0; i < items.size(); ++i) row.push_back(valueAt(i, (int64_t)r));
      } else {
        std::vector<std::string>& row = res.rows[r];
        row.reserve(items.size());
        for (size_t i = 0; i < items.size(); ++i) row.push_back(cell(valueAt(i, (int64_t)r)));
      }
    }
  }, 4096);
  pproj.done(cnt);
  res.info = ctx.tabs.size() == 1 ? "gpu-scan" : "gpu-join";
  if (envSetting("TRACE")) fprintf(stderr, "[trace] %.2f ms %s\n", msSince(t0), res.info.c_str());
  return res;
}

}  // namespace gdb

