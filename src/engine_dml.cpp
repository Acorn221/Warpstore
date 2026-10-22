#include "engine_core.h"

namespace gdb {
// ------------------------------------------------------------------ DELETE / UPDATE

// UPDATE / DELETE with joins (UPDATE a JOIN b .. SET, DELETE a FROM a JOIN b ..), correlated
// subqueries, ORDER BY / LIMIT: the rows to change (and their new values) come from a SELECT over
// the same FROM / WHERE, then the target rows with exactly those column values are modified (rows
// identical in every column are interchangeable, so matching by value is exact).
ResultSet Engine::modifyViaSelect(Stmt& s) {
  auto t = cat_.get(s.table);
  Stmt sel;
  sel.type = Stmt::SELECT;
  sel.from = s.from;
  if (sel.from.empty()) sel.from = {TableRef{s.table, plainName(s.table), nullptr, 0, nullptr}};
  sel.joinConds = s.joinConds;
  sel.where = s.where;
  sel.orderBy = s.orderBy;
  sel.limit = s.limit;
  const std::string alias = s.targetAlias.empty() ? sel.from[0].alias : s.targetAlias;
  for (auto& c : t->cols) sel.items.push_back({mkCol(alias, c.name), "__c" + std::to_string(sel.items.size())});
  if (s.limit >= 0 && sel.orderBy.empty())  // LIMIT without ORDER BY: rows chosen by value, so a WAL replay picks the same ones
    for (auto& c : t->cols)
      if (c.pt != PType::VEC) sel.orderBy.push_back(OrderItem{mkCol(alias, c.name), false});
  std::vector<int> setCols;
  for (auto& [name, e] : s.sets) {
    const int c = t->colIndex(name);
    if (c < 0) throw SqlError("Unknown column '" + name + "' in 'field list'", 1054);
    setCols.push_back(c);
    sel.items.push_back({cloneExpr(e), "__s" + std::to_string(sel.items.size())});
  }
  QueryCtx q;
  ResultSet r;
  try {
    r = query(sel, q, false, true);
  } catch (...) {
    dropTemps(q);
    throw;
  }
  dropTemps(q);
  const size_t nc = t->cols.size();
  auto keyOf = [&](const std::vector<Value>& row) {
    std::string k;
    for (size_t c = 0; c < nc; ++c) {
      const Value& v = row[c];
      if (v.isNull()) { k += "\x02\x1f"; continue; }
      if (v.kind == Value::DBL) {
        char b[40];
        snprintf(b, sizeof b, "%.15g", v.d);
        k += b;
      } else if (v.kind == Value::STR) {
        k += v.s;
      } else {
        k += std::to_string(v.i);
      }
      k += '\x1f';
    }
    return k;
  };
  std::unordered_map<std::string, size_t> want;  // row values -> result row (its SET values)
  for (size_t i = 0; i < r.vals.size(); ++i) want.emplace(keyOf(r.vals[i]), i);
  std::unique_lock<FairSharedMutex> lk(t->mu);
  std::vector<Loc> matched;
  std::vector<std::vector<Value>> newRows;
  std::unordered_map<std::string, int64_t> budget;  // LIMIT: how many rows with these values may change
  if (s.limit >= 0)
    for (auto& row : r.vals) ++budget[keyOf(row)];
  forEachLiveRow(*t, [&](Loc l) {
    std::vector<Value> cur;
    for (size_t c = 0; c < nc; ++c) cur.push_back(cellValue(*t, l, (int)c));
    const std::string k = keyOf(cur);
    auto it = want.find(k);
    if (it == want.end()) return;
    if (s.limit >= 0 && budget[k]-- <= 0) return;
    matched.push_back(l);
    if (s.type == Stmt::UPDATE) {
      std::vector<Value> row = cur;
      for (size_t j = 0; j < setCols.size(); ++j) {
        Value v = r.vals[it->second][nc + j];
        if (!v.isNull() && t->cols[setCols[j]].lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
        row[setCols[j]] = v;
      }
      newRows.push_back(std::move(row));
    }
  });
  if (rowsPending(*t, matched)) {  // another open transaction changed some of these rows: wait, then redo
    waitTxnFree(*t, lk, &matched);
    lk.unlock();
    return modifyViaSelect(s);
  }
  return applyModify(*t, matched, newRows);
}

static void markDeleted(Table* t, const std::vector<Loc>& matched);

// Before an UPDATE deletes anything: the new values break no PRIMARY / UNIQUE key (checked on the
// statement's end state; MySQL checks row by row, so `SET id = id + 1` it may refuse is accepted).
// Only keys with a SET column: the others keep their values. setI / setD: per SET column, the new
// raw values of the matched rows, in order.
static void checkUniqueAfterUpdate(Table& t, const std::vector<Loc>& matched, const std::vector<int>& setCols,
                                   const std::vector<std::vector<int64_t>>& setI, const std::vector<std::vector<double>>& setD) {
  std::vector<std::pair<std::vector<int>, const IndexMeta*>> keys;  // (columns, index; null: the primary key)
  auto touches = [&](const std::vector<int>& cs) {
    for (int c : cs)
      if (std::find(setCols.begin(), setCols.end(), c) != setCols.end()) return true;
    return false;
  };
  if (t.pk >= 0 && touches({t.pk})) keys.push_back({{t.pk}, nullptr});
  for (auto& ix : t.indexes)
    if (ix.unique && !ix.vec && touches(ix.cols)) keys.push_back({ix.cols, &ix});
  if (keys.empty() || matched.empty()) return;
  std::unordered_set<uint64_t> mine;  // the matched rows: their old keys are free
  for (const Loc& l : matched) mine.insert(TxnPending::key(l));
  auto raw = [&](size_t i, int c, int64_t& k) {  // new value of column c of matched row i (false: NULL)
    const size_t j = std::find(setCols.begin(), setCols.end(), c) - setCols.begin();
    if (j == setCols.size()) return t.rawKey(matched[i], c, k);
    if (t.cols[c].pt == PType::F64) {
      double d = setD[j][i];
      if (d != d) return false;
      if (d == 0) d = 0;
      memcpy(&k, &d, 8);
      return true;
    }
    k = setI[j][i];
    return k != (t.cols[c].pt == PType::I32 ? (int64_t)kNullI32 : kNullI64);
  };
  for (auto& [cs, ix] : keys) {
    std::set<std::vector<int64_t>> seen;
    std::vector<int64_t> tup(cs.size());
    for (size_t i = 0; i < matched.size(); ++i) {
      bool null = false;
      for (size_t q = 0; q < cs.size() && !null; ++q) null = !raw(i, cs[q], tup[q]);
      if (null) continue;  // (NULLs never collide; a NULL primary key is refused when it is written)
      const std::string what = ix ? "Duplicate entry for key '" + ix->name + "'" : "Duplicate entry '" + std::to_string(tup[0]) + "' for key 'PRIMARY'";
      if (!seen.insert(tup).second) throw SqlError(what, 1062);
      if (!ix) {
        int64_t key = tup[0];
        if (t.cols[t.pk].pt == PType::F64) {  // (the primary-key index holds a DOUBLE key as an integer)
          double d;
          memcpy(&d, &tup[0], 8);
          key = (int64_t)d;
        }
        auto it = t.pkIndex.find(key);
        if (it != t.pkIndex.end() && !mine.count(TxnPending::key(it->second))) throw SqlError(what, 1062);
        continue;
      }
      auto it = ix->map.find(tup[0]);
      if (it == ix->map.end()) continue;
      for (const Loc& l : it->second) {
        if (mine.count(TxnPending::key(l))) continue;
        bool same = true;
        for (size_t q = 0; q < cs.size() && same; ++q) {
          int64_t a;
          same = t.rawKey(l, cs[q], a) && a == tup[q];
        }
        if (same) throw SqlError(what, 1062);
      }
    }
  }
}

// UPDATE outside a transaction, without VECTOR columns: each new row starts as the old row's stored
// cells (strings stay codes); Values are made only for the columns the SET expressions read, and
// only the SET columns are converted back. Matched locations stay valid while the new rows are
// appended (full write buffers are flushed at the end).
static ResultSet updateRaw(Table& t, const std::vector<Loc>& matched, const std::vector<std::pair<int, ExprP>>& sets,
                           const std::set<int>& reads) {
  const bool tr = envSetting("TRACE") != nullptr;
  auto ph = Clock::now();
  double tg = 0, ts = 0, ta = 0;
  const size_t nc = t.cols.size();
  // Column-level UPDATE: rows in segments keep their places and only the SET columns of the segments
  // they are in get new versions (Table::replaceColumns), when no SET column is a key, the sort key,
  // indexed or AUTO_INCREMENT (their indexes / order would change). Rows in write buffers (and all
  // rows otherwise) are deleted and re-inserted.
  std::vector<int> setCols;  // (distinct, in order)
  for (auto& [c, e] : sets)
    if (std::find(setCols.begin(), setCols.end(), c) == setCols.end()) setCols.push_back(c);
  bool colMode = !txnKey() && !t.segs.empty();
  {
    std::set<int> fixed{t.pk, t.sortKey, t.autoIncCol};
    for (int k : t.shardKey) fixed.insert(k);
    for (auto& ix : t.indexes)
      for (int c : ix.cols) fixed.insert(c);
    for (int c : setCols) colMode = colMode && !fixed.count(c) && t.cols[c].pt != PType::VEC;
  }
  // Per segment: in place when enough of its rows match. A new column version costs the segment's
  // rows x the SET columns' width (copied, uploaded); moving costs the matched rows x the row width,
  // about 4 times over (the append, then compaction rewriting what is left).
  std::vector<uint8_t> segIn(t.segs.size(), 0);  // segments whose rows are updated in place
  std::vector<int64_t> cnt(t.segs.size(), 0);      // matched rows per segment
  if (colMode) {
    for (const Loc& l : matched)
      if (l.seg >= 0) ++cnt[l.seg];
    int64_t rowBytes = 0, setBytes = 0;
    for (size_t c = 0; c < nc; ++c) rowBytes += t.colBytes((int)c);
    for (int c : setCols) setBytes += t.colBytes(c);
    for (size_t i = 0; i < t.segs.size(); ++i)
      segIn[i] = cnt[i] > 0 && t.segs[i]->full() && cnt[i] * rowBytes * 4 >= t.segs[i]->n * setBytes;
  }
  auto inPlace = [&](const Loc& l) { return l.seg >= 0 && segIn[l.seg]; };
  // The new versions of the SET columns of segments with matched rows: per segment (slotOf) and SET
  // column, a copy of the column with the new values written in; installed at the end.
  std::vector<int32_t> slotOf(t.segs.size(), -1), psegs;
  std::vector<Table::ColPatch> pdata;
  std::vector<Loc> movedRows;  // deleted and re-inserted (after all the new values are known)
  const std::vector<Loc>* moved = &matched;
  if (colMode) {
    moved = &movedRows;
    for (const Loc& l : matched) {
      if (!inPlace(l)) movedRows.push_back(l);
      else if (slotOf[l.seg] < 0) slotOf[l.seg] = (int32_t)psegs.size(), psegs.push_back(l.seg);
    }
    pdata.resize(psegs.size());
    for (auto& p : pdata) p.cols.resize(setCols.size()), p.vals.resize(setCols.size());
    for (int32_t si : psegs)  // (cold columns loaded here: a pool task must not wait for the I/O pool)
      for (int c : setCols) t.colData(Loc{si, 0}, c);
    ioParallel(psegs.size() * setCols.size(), [&](size_t i) {
      const size_t s = i / setCols.size(), k = i % setCols.size();
      pdata[s].cols[k] = t.colData(Loc{psegs[s], 0}, setCols[k]);
      pdata[s].vals[k].resize((size_t)cnt[psegs[s]] * t.colBytes(setCols[k]));
      if (k == 0) pdata[s].rows.resize((size_t)cnt[psegs[s]]);
    }, true);
  }
  if (tr) fprintf(stderr, "[trace] update: column copies %.1f ms\n", msSince(ph));
  t.deferFlush = true;
  struct Flush {
    Table& t;
    ~Flush() {
      t.deferFlush = false;
      try {
        for (int p = 0; p < t.nparts; ++p)
          if (t.bufs[p]->n >= kSegRows) t.flushPart(p);
      } catch (std::exception& e) {
        fprintf(stderr, "[warpstore] flush after UPDATE: %s\n", e.what());
      }
    }
  } flush{t};
  std::vector<Value> cur(nc), vals(sets.size());
  PostCtx pc;
  pc.row = &cur;
  pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
  // In chunks: the matched rows' cells, a column per task; the SET values, row by row; then the new
  // rows appended a column at a time.
  const size_t kChunk = colMode ? (1u << 20) : 262144;
  std::vector<size_t> cursor(psegs.size(), 0);  // per segment: its rows / vals written so far
  std::vector<std::vector<int64_t>> iv(nc);
  std::vector<std::vector<double>> dv(nc);
  // Fused: every matched row is updated in place and every SET is a constant, a copy of another
  // column, or a column +/- a constant: the new values are computed while the segments' column copies
  // are patched, from the old values (no gather, no value arrays). Anything else (an overflow, a
  // NULL into NOT NULL) goes the general way below.
  bool fused = false;
  if (colMode && !psegs.empty() && sets.size() == setCols.size()) {
    struct Form {
      int kind = -1;  // 0 constant, 1 copy of src, 2 src + add (ints) / addD (doubles)
      int src = -1;
      int64_t i0 = 0, add = 0;
      double d0 = 0, addD = 0;
    };
    std::vector<Form> forms(sets.size());
    bool ok = true;
    for (const Loc& l : matched) ok = ok && inPlace(l);
    std::set<int> assigned;
    for (size_t j = 0; j < sets.size() && ok; ++j) {
      const auto& [c, e] = sets[j];
      const ColumnMeta& cm = t.cols[c];
      auto colOf = [&](const ExprP& x) {
        return x->kind == Expr::COL && x->bt == 0 && x->bc >= 0 && t.cols[x->bc].lt == cm.lt && t.cols[x->bc].pt == cm.pt &&
                       t.cols[x->bc].scale == cm.scale && cm.lt != LType::VARCHAR ? x->bc : -1;
      };
      Form& f = forms[j];
      if (e->kind == Expr::LIT && !e->lit.isNull() && (cm.lt != LType::VARCHAR || e->lit.kind == Value::STR)) {
        f.kind = 0;
        t.rawOf(c, e->lit, f.i0, f.d0);
      } else if (const int s2 = colOf(e); s2 >= 0) {
        f.kind = 1, f.src = s2;
      } else if (e->kind == Expr::BIN && (e->op == "+" || e->op == "-") && colOf(e->args[0]) >= 0 && e->args[1]->kind == Expr::LIT &&
                 !e->args[1]->lit.isNull() && e->args[1]->lit.kind != Value::STR &&
                 (cm.lt == LType::INT || cm.lt == LType::BIGINT || cm.lt == LType::DECIMAL || cm.lt == LType::DOUBLE)) {
        f.kind = 2, f.src = colOf(e->args[0]);
        const double lv = e->args[1]->lit.num() * (e->op == "-" ? -1 : 1);
        if (cm.pt == PType::F64) {
          f.addD = lv;
        } else {
          const double scaled = cm.lt == LType::DECIMAL ? lv * pow10i(cm.scale) : lv;
          f.add = (int64_t)std::llround(scaled);
          ok = std::fabs(scaled - (double)f.add) < 1e-9 && std::fabs(scaled) < 9e15;
        }
      } else {
        ok = false;
      }
      if (f.src >= 0 && assigned.count(f.src)) ok = false;  // (reads a column an earlier SET assigned)
      assigned.insert(c);
    }
    if (ok) {
      struct Run {
        uint32_t s;
        size_t b, e, off;
      };
      std::vector<Run> runs;
      for (size_t i = 0; i < matched.size(); ++i) {
        const uint32_t sl = (uint32_t)slotOf[matched[i].seg];
        if (!runs.empty() && runs.back().s == sl && runs.back().e == i && runs.back().e - runs.back().b < 16384) ++runs.back().e;
        else runs.push_back({sl, i, i + 1, cursor[sl]});
        ++cursor[sl];
      }
      std::atomic<bool> fail{false};
      const size_t K = setCols.size();
      for (int32_t si : psegs)  // (source columns loaded here: a pool task must not wait for the I/O pool)
        for (const Form& f : forms)
          if (f.src >= 0) t.colData(Loc{si, 0}, f.src);
      ioParallel(runs.size() * K, [&](size_t x) {
        const Run& ru = runs[x / K];
        const size_t k = x % K;
        const int c = setCols[k];
        const ColumnMeta& cm = t.cols[c];
        const Form& f = forms[k];
        Table::ColPatch& p = pdata[ru.s];
        const int32_t seg = psegs[ru.s];
        uint8_t* col = p.cols[k].data();
        const uint8_t* src = f.src >= 0 ? t.colData(Loc{seg, 0}, f.src).data() : nullptr;  // (old values: present, loaded above)
        const bool notNull = cm.notNull || c == t.pk;
        const int64_t nul = cm.pt == PType::I64 ? kNullI64 : kNullI32;
        const int64_t lo = cm.pt == PType::I64 ? INT64_MIN + 1 : INT32_MIN + 1, hi = cm.pt == PType::I64 ? INT64_MAX : INT32_MAX;
        uint8_t* vals = p.vals[k].data() + ru.off * (size_t)t.colBytes(c);
        for (size_t i = ru.b; i < ru.e; ++i) {
          const int32_t r = matched[i].row;
          if (k == 0) p.rows[ru.off + (i - ru.b)] = r;
          if (cm.pt == PType::F64) {
            double v = f.d0;
            if (f.kind) {
              memcpy(&v, src + (size_t)r * 8, 8);
              if (v != v) {
                if (notNull) return (void)(fail = true);
              } else {
                v += f.addD;
              }
            }
            memcpy(col + (size_t)r * 8, &v, 8), memcpy(vals, &v, 8), vals += 8;
            continue;
          }
          int64_t v = f.i0;
          if (f.kind) {
            if (cm.pt == PType::I64) memcpy(&v, src + (size_t)r * 8, 8);
            else { int32_t x32; memcpy(&x32, src + (size_t)r * 4, 4); v = x32; }
            if (v == nul) {
              if (notNull) return (void)(fail = true);
            } else if (f.kind == 2 && (__builtin_add_overflow(v, f.add, &v) || v < lo || v > hi)) {
              return (void)(fail = true);  // (out of range: the general path reports it)
            }
          }
          if (cm.pt == PType::I64) memcpy(col + (size_t)r * 8, &v, 8), memcpy(vals, &v, 8), vals += 8;
          else { const int32_t x32 = (int32_t)v; memcpy(col + (size_t)r * 4, &x32, 4), memcpy(vals, &x32, 4), vals += 4; }
        }
      }, true);
      if (!fail) {
        fused = true;
      } else {  // (back to the old columns, then the general path)
        std::fill(cursor.begin(), cursor.end(), 0);
        ioParallel(psegs.size() * K, [&](size_t i) {
          const size_t s = i / K, k = i % K;
          pdata[s].cols[k] = t.colData(Loc{psegs[s], 0}, setCols[k]);
        }, true);
      }
    }
    if (tr && fused) fprintf(stderr, "[trace] update: fused in place, %.1f ms\n", msSince(ph));
  }
  // Two passes: all the new values first (an error, e.g. a NULL into NOT NULL, leaves the table as
  // it was), then the rows deleted and the new ones written.
  std::vector<std::vector<int64_t>> setI(setCols.size());
  std::vector<std::vector<double>> setD(setCols.size());
  for (int pass = fused ? 2 : 0; pass < 2; ++pass) {
  if (pass == 1) {
    checkUniqueAfterUpdate(t, matched, setCols, setI, setD);
    markDeleted(&t, *moved);
  }
  for (size_t from = 0; from < matched.size(); from += kChunk) {
    const size_t n = std::min(kChunk, matched.size() - from);
    const Loc* ls = matched.data() + from;
    auto gather = [&](size_t c, size_t from, size_t to) {  // (arrays sized already)
      const PType pt = t.cols[c].pt;
      const uint8_t* base = nullptr;
      int32_t baseSeg = INT32_MIN;
      for (size_t i = from; i < to; ++i) {
        const Loc l = ls[i];
        if (l.seg != baseSeg) base = t.colData(l, (int)c).data(), baseSeg = l.seg;
        switch (pt) {
          case PType::I32: iv[c][i] = ((const int32_t*)base)[l.row]; break;
          case PType::I64: iv[c][i] = ((const int64_t*)base)[l.row]; break;
          default: dv[c][i] = ((const double*)base)[l.row]; break;
        }
      }
    };
    ph = Clock::now();
    // Tables without keys / indexes / open transactions: the new rows are copied from the old ones
    // straight into the write buffers (appendCopy), so only the columns the SETs read are gathered.
    bool shardSet = false;
    for (auto& [c, e] : sets)
      for (int k : t.shardKey) shardSet = shardSet || c == k;
    const bool copyRows = t.pk < 0 && t.indexes.empty() && !txnKey() && !shardSet;
    size_t nMoved = n;  // rows deleted and re-inserted
    if (colMode) {
      nMoved = 0;
      for (size_t i = 0; i < n; ++i) nMoved += !inPlace(ls[i]);
    }
    // first pass: the columns the SETs read; second: the rest of the moved rows (copied straight from
    // the old rows without keys / indexes)
    const bool sparse = pass == 0 || copyRows || nMoved == 0;
    std::vector<uint8_t> need(nc, sparse ? 0 : 1);
    if (sparse) {
      std::set<std::pair<int, int>> used;
      for (auto& [c, e] : sets) colsOf(e, used);
      for (auto& [bt, bc] : used)
        if (pass == 0 && bt == 0 && bc >= 0 && bc < (int)nc) need[bc] = 1;
      for (auto& [c, e] : sets) {  // (the SET columns' value arrays)
        if (t.cols[c].pt == PType::F64) dv[c].resize(n);
        else iv[c].resize(n);
      }
    }
    std::vector<size_t> gcols;
    for (size_t c = 0; c < nc; ++c)
      if (need[c]) gcols.push_back(c);
    for (size_t c : gcols) {
      if (t.cols[c].pt == PType::F64) dv[c].resize(n);
      else iv[c].resize(n);
    }
    if (n < 4096) {  // (a few rows, e.g. by primary key: not worth a task per column)
      for (size_t c : gcols) gather(c, 0, n);
    } else {
      // cold columns (disk tier) loaded here first: a pool task must not wait for the I/O pool
      std::vector<uint8_t> segsSeen(t.segs.size(), 0);
      int32_t last = INT32_MIN;
      for (size_t i = 0; i < n; ++i)
        if (ls[i].seg != last && (last = ls[i].seg) >= 0 && !segsSeen[last]++)
          for (size_t c : gcols) t.colData(ls[i], (int)c);
      // a task per column and block of rows (few columns: several blocks each)
      const size_t nb = gcols.empty() ? 1 : std::max<size_t>(1, 16 / gcols.size()), bs = (n + nb - 1) / nb;
      ioParallel(gcols.size() * nb, [&](size_t k) {
        const size_t b = k % nb;
        gather(gcols[k / nb], b * bs, std::min(n, (b + 1) * bs));
      }, true);
    }
    tg += msSince(ph), ph = Clock::now();
    if (pass == 1) {  // (the values from the first pass)
      for (size_t k = 0; k < setCols.size(); ++k) {
        const int c = setCols[k];
        if (t.cols[c].pt == PType::F64) dv[c].assign(setD[k].begin() + from, setD[k].begin() + from + n);
        else iv[c].assign(setI[k].begin() + from, setI[k].begin() + from + n);
      }
    } else {
    // The SET values, row by row: on all cores unless a string column is set (new strings are coded in
    // row order, as a replay of the statement codes them).
    // MySQL evaluates single-table SETs left to right: a SET that reads a column an earlier one assigns
    // sees the new value (then everything goes row by row, in order).
    bool chained = false;
    {
      std::set<int> assigned;
      for (auto& [c, e] : sets) {
        std::set<std::pair<int, int>> used;
        colsOf(e, used);
        for (auto& [bt, bc] : used) chained = chained || (bt == 0 && assigned.count(bc));
        assigned.insert(c);
      }
    }
    std::vector<std::pair<int, ExprP>> rs = sets;  // the SETs evaluated row by row
    auto setRows = [&](size_t b, size_t e, std::vector<Value>& cur, std::vector<Value>& vals, PostCtx& pc) {
      int64_t i0;
      double d0;
      for (size_t i = b; i < e; ++i) {
        for (int c : reads) cur[c] = cellValue(t, ls[i], c);
        for (size_t j = 0; j < rs.size(); ++j) {  // (left to right: a later SET reads the new values)
          vals[j] = evalPost(rs[j].second, pc);
          if (chained) cur[rs[j].first] = vals[j];
        }
        for (size_t j = 0; j < rs.size(); ++j) {
          const int c = rs[j].first;
          Value& v = vals[j];
          if (!v.isNull() && t.cols[c].lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
          t.rawOf(c, v, i0, d0);
          if (t.cols[c].pt == PType::F64) dv[c][i] = d0;
          else iv[c][i] = i0;
        }
      }
    };
    // Vectorized SET forms on the raw values (computed against the old row first): a constant,
    // another column, column +/- constant. Others (or an overflow) go row by row.
    std::vector<int> fastC;
    std::vector<std::vector<int64_t>> fi;
    std::vector<std::vector<double>> fd;
    rs.clear();
    for (auto& [c, e] : sets) {
      if (chained) {
        rs.push_back({c, e});
        continue;
      }
      const ColumnMeta& cm = t.cols[c];
      std::vector<int64_t> oi;
      std::vector<double> od;
      bool ok = false;
      const bool f64 = cm.pt == PType::F64;
      auto colOf = [&](const ExprP& x) {
        return x->kind == Expr::COL && x->bt == 0 && x->bc >= 0 && t.cols[x->bc].lt == cm.lt && t.cols[x->bc].pt == cm.pt &&
                       t.cols[x->bc].scale == cm.scale && cm.lt != LType::VARCHAR ? x->bc : -1;
      };
      if (e->kind == Expr::LIT && !e->lit.isNull() && (cm.lt != LType::VARCHAR || e->lit.kind == Value::STR)) {
        int64_t i0 = 0;  // (a string constant: coded once)
        double d0 = 0;
        t.rawOf(c, e->lit, i0, d0);
        if (f64) od.assign(n, d0);
        else oi.assign(n, i0);
        ok = true;
      } else if (const int s2 = colOf(e); s2 >= 0) {
        if (f64) od = dv[s2];
        else oi = iv[s2];
        ok = true;
      } else if (e->kind == Expr::BIN && (e->op == "+" || e->op == "-") && colOf(e->args[0]) >= 0 && e->args[1]->kind == Expr::LIT &&
                 !e->args[1]->lit.isNull() && e->args[1]->lit.kind != Value::STR &&
                 (cm.lt == LType::INT || cm.lt == LType::BIGINT || cm.lt == LType::DECIMAL || cm.lt == LType::DOUBLE)) {
        const int s2 = colOf(e->args[0]);
        const double lv = e->args[1]->lit.num() * (e->op == "-" ? -1 : 1);
        if (f64) {
          od.resize(n);
          parallelFor(n, [&](size_t b, size_t e) {
            for (size_t i = b; i < e; ++i) od[i] = dv[s2][i] == dv[s2][i] ? dv[s2][i] + lv : dv[s2][i];
          }, 65536);
          ok = true;
        } else {
          const double scaled = cm.lt == LType::DECIMAL ? lv * pow10i(cm.scale) : lv;
          const int64_t add = (int64_t)std::llround(scaled);
          const int64_t nul = cm.pt == PType::I64 ? kNullI64 : kNullI32;
          const int64_t lo = cm.pt == PType::I64 ? INT64_MIN + 1 : INT32_MIN + 1, hi = cm.pt == PType::I64 ? INT64_MAX : INT32_MAX;
          if (std::fabs(scaled - (double)add) < 1e-9 && std::fabs(scaled) < 9e15) {
            oi.resize(n);
            std::atomic<bool> over{false};
            parallelFor(n, [&](size_t b, size_t e) {
              for (size_t i = b; i < e; ++i) {
                const int64_t x = iv[s2][i];
                int64_t y = x;
                if (x != nul && (__builtin_add_overflow(x, add, &y) || y < lo || y > hi)) {
                  over = true;
                  return;
                }
                oi[i] = y;
              }
            }, 65536);
            ok = !over;
          }
        }
      }
      if (ok && (cm.notNull || c == t.pk)) {  // a NULL into NOT NULL: the row path reports it
        const int64_t nul = cm.pt == PType::I64 ? kNullI64 : kNullI32;
        for (int64_t x : oi) ok = ok && x != nul;
        for (double x : od) ok = ok && x == x;
      }
      if (ok) {
        fastC.push_back(c);
        fi.push_back(std::move(oi));
        fd.push_back(std::move(od));
      } else {
        rs.push_back({c, e});
      }
    }
    vals.resize(rs.size());
    bool strSet = false;
    for (auto& [c, e] : rs) strSet = strSet || t.cols[c].lt == LType::VARCHAR;
    if (rs.empty()) {
    } else if (n < 16384) {
      setRows(0, n, cur, vals, pc);
    } else if (strSet) {
      // Values on all cores; strings then coded in one batch in row order (first appearance), as the
      // row-by-row path and a replay code them.
      const size_t R = rs.size();
      std::vector<Value> vb(n * R);
      parallelFor(n, [&](size_t b, size_t e) {
        std::vector<Value> lcur(nc);
        PostCtx lpc = pc;
        lpc.row = &lcur;
        for (size_t i = b; i < e; ++i) {
          for (int c : reads) lcur[c] = cellValue(t, ls[i], c);
          for (size_t j = 0; j < R; ++j) {
            Value v = evalPost(rs[j].second, lpc);
            if (!v.isNull() && t.cols[rs[j].first].lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
            if (chained) lcur[rs[j].first] = v;
            vb[i * R + j] = std::move(v);
          }
        }
      }, 4096);
      for (size_t j = 0; j < R; ++j) {
        const int c = rs[j].first;
        if (t.cols[c].lt == LType::VARCHAR) {
          std::vector<std::string_view> sv(n);
          bool anyNull = false;
          for (size_t i = 0; i < n; ++i) {
            const Value& v = vb[i * R + j];
            if (v.isNull()) anyNull = true;
            else sv[i] = std::string_view(v.s.data() ? v.s.data() : "", v.s.size());
          }
          int64_t nullCode = kNullI32;
          if (anyNull) {  // (the NOT NULL check and the NULL code, as row by row)
            double d0;
            t.rawOf(c, Value::Null(), nullCode, d0);
          }
          t.dicts[c]->getOrAddMany(sv.data(), n, iv[c].data(), nullCode);
        } else {
          parallelFor(n, [&](size_t b, size_t e) {
            int64_t i0;
            double d0;
            for (size_t i = b; i < e; ++i) {
              t.rawOf(c, vb[i * R + j], i0, d0);
              if (t.cols[c].pt == PType::F64) dv[c][i] = d0;
              else iv[c][i] = i0;
            }
          }, 4096);
        }
      }
    } else {
      parallelFor(n, [&](size_t b, size_t e) {
        std::vector<Value> lcur(nc), lvals(sets.size());
        PostCtx lpc = pc;
        lpc.row = &lcur;
        setRows(b, e, lcur, lvals, lpc);
      }, 4096);
    }
    for (size_t j = 0; j < fastC.size(); ++j) {
      if (t.cols[fastC[j]].pt == PType::F64) dv[fastC[j]].swap(fd[j]);
      else iv[fastC[j]].swap(fi[j]);
    }
    for (size_t k = 0; k < setCols.size(); ++k) {
      const int c = setCols[k];
      if (t.cols[c].pt == PType::F64) setD[k].insert(setD[k].end(), dv[c].begin(), dv[c].begin() + n);
      else setI[k].insert(setI[k].end(), iv[c].begin(), iv[c].begin() + n);
    }
    ts += msSince(ph);
    continue;
    }
    ts += msSince(ph), ph = Clock::now();
    // The rows deleted above, re-inserted with their new values (iv / dv: theirs, in order).
    auto moveRest = [&](const Loc* ls, size_t n) {
      if (copyRows) {
        std::vector<int32_t> part(n);
        for (size_t i = 0; i < n; ++i) part[i] = ls[i].seg >= 0 ? t.segs[ls[i].seg]->part : -1 - ls[i].seg;
        std::vector<uint8_t> given(nc, 0);
        for (auto& [c, e] : sets) given[c] = 1;
        t.appendCopy(ls, n, part.data(), given, iv, dv);
        return;
      }
      // Rows stay in their partition when no shard-key column is set (no re-hashing).
      bool keepPart = true;
      for (auto& [c, e] : sets)
        for (int k : t.shardKey) keepPart = keepPart && c != k;
      std::vector<int32_t> partOf;
      if (keepPart && t.nparts > 1 && !t.shardKey.empty()) {
        partOf.resize(n);
        for (size_t i = 0; i < n; ++i) partOf[i] = ls[i].seg >= 0 ? t.segs[ls[i].seg]->part : -1 - ls[i].seg;
      }
      t.appendColumns(iv, dv, (int64_t)n, partOf.empty() ? nullptr : partOf.data());
    };
    if (nMoved < n) {
      // Rows in segments: their new values go into the segments' new column versions (and are kept,
      // with their rows, for the device copies).
      struct Run {  // chunk rows [b, e) are rows of segment slot s; they go at off in its rows / vals
        uint32_t s, b, e;
        size_t off;
      };
      std::vector<Run> runs;
      for (size_t i = 0; i < n; ++i) {
        if (!inPlace(ls[i])) continue;
        const uint32_t sl = (uint32_t)slotOf[ls[i].seg];
        if (!runs.empty() && runs.back().s == sl && runs.back().e == i && runs.back().e - runs.back().b < 16384) ++runs.back().e;
        else runs.push_back({sl, (uint32_t)i, (uint32_t)i + 1, cursor[sl]});
        ++cursor[sl];
      }
      const size_t K = setCols.size();
      ioParallel(runs.size() * K, [&](size_t x) {  // (a task per run of at most 16K rows and SET column)
        const Run& ru = runs[x / K];
        const size_t k = x % K;
        const int c = setCols[k];
        const int w = t.colBytes(c);
        Table::ColPatch& p = pdata[ru.s];
        uint8_t* col = p.cols[k].data();
        uint8_t* vals = p.vals[k].data() + ru.off * w;
        const bool f64 = t.cols[c].pt == PType::F64;
        for (uint32_t i = ru.b; i < ru.e; ++i, vals += w) {
          const int32_t r = ls[i].row;
          if (k == 0) p.rows[ru.off + (i - ru.b)] = r;
          if (w == 4) {
            const int32_t v = (int32_t)iv[c][i];
            memcpy(col + (size_t)r * 4, &v, 4);
            memcpy(vals, &v, 4);
          } else {
            const void* v = f64 ? (const void*)&dv[c][i] : (const void*)&iv[c][i];
            memcpy(col + (size_t)r * 8, v, 8);
            memcpy(vals, v, 8);
          }
        }
      }, true);
      if (nMoved == 0) {
        ta += msSince(ph);
        continue;
      }
      // The rest (write buffer rows) moves: the chunk is narrowed to them.
      std::vector<Loc> keep;
      keep.reserve(nMoved);
      size_t o = 0;
      for (size_t i = 0; i < n; ++i) {
        if (inPlace(ls[i])) continue;
        keep.push_back(ls[i]);
        for (size_t c = 0; c < nc; ++c) {
          if (iv[c].size() == n) iv[c][o] = iv[c][i];
          if (dv[c].size() == n) dv[c][o] = dv[c][i];
        }
        ++o;
      }
      for (size_t c = 0; c < nc; ++c) {
        if (iv[c].size() == n) iv[c].resize(o);
        if (dv[c].size() == n) dv[c].resize(o);
      }
      moveRest(keep.data(), o);
      ta += msSince(ph);
      continue;
    }
    moveRest(ls, n);
    ta += msSince(ph);
  }
  }
  if (!psegs.empty()) {
    ph = Clock::now();
    t.replaceColumns(psegs, setCols, pdata);
    if (tr) fprintf(stderr, "[trace] update: %zu segments' columns replaced in %.1f ms\n", psegs.size(), msSince(ph));
  }
  if (tr) fprintf(stderr, "[trace] update: gather %.1f ms, set %.1f ms, append %.1f ms\n", tg, ts, ta);
  return okResult((int64_t)matched.size(), "Rows matched: " + std::to_string(matched.size()));
}

ResultSet Engine::doModify(Stmt& s) {
  std::function<bool(const ExprP&)> hasSub = [&](const ExprP& e) {
    if (!e) return false;
    if (e->sub || e->kind == Expr::SUBQUERY) return true;
    for (auto& a : e->args)
      if (hasSub(a)) return true;
    return false;
  };
  bool viaSelect = s.from.size() > 1 || !s.joinConds.empty() || s.limit >= 0 || hasSub(s.where) ||
                   (s.from.size() == 1 && (s.from[0].sub || strcasecmp(s.from[0].alias.c_str(), plainName(s.from[0].name).c_str()) != 0));
  for (auto& [n, e] : s.sets) viaSelect = viaSelect || hasSub(e);
  if (viaSelect) return modifyViaSelect(s);
  // Subqueries in WHERE / SET run first (they may read this table).
  QueryCtx q;
  {
    Stmt tmp;
    tmp.where = s.where;
    for (auto& [n, e] : s.sets) tmp.items.push_back({e, n});
    try {
      resolveSubqueries(tmp, q);
    } catch (...) {
      dropTemps(q);
      throw;
    }
    s.where = tmp.where;
    for (size_t i = 0; i < s.sets.size(); ++i) s.sets[i].second = tmp.items[i].e;
    dropTemps(q);
  }
  auto t = cat_.get(s.table);
  std::unique_lock<FairSharedMutex> lk(t->mu);
  BindCtx ctx;
  ctx.tabs = {t};
  ctx.aliases = {t->name};
  if (s.where) bind(s.where, ctx);
  std::vector<std::pair<int, ExprP>> sets;
  ctx.rowWise = true;
  for (auto& [name, e] : s.sets) {
    int c = t->colIndex(name);
    if (c < 0) throw SqlError("Unknown column '" + name + "' in 'field list'", 1054);
    bind(e, ctx);
    sets.push_back({c, e});
  }
  std::vector<ExprP> conj;
  flattenAnd(s.where, conj);
  Compiler comp(&ctx);
  Prog wp;
  if (s.where) wp = comp.compile(s.where);

  ExprEnv henv = comp.hostEnv();
  auto hostCols = [&](int seg) {
    for (size_t k = 0; k < comp.slotCols.size(); ++k) {
      int c = comp.slotCols[k].second;
      henv.cols[k] = comp.env.cols[k];
      henv.cols[k].p = t->colData(Loc{(int32_t)seg, 0}, c).data();
      henv.cols[k].enc = 0;  // host mirror is raw
    }
  };

  const auto modT0 = Clock::now();
  std::vector<Loc> matched;
  int64_t key;
  bool ixFound = false;
  const IndexMeta* ixm = nullptr;
  if (findPkEq(conj, *t, 0, key)) {
    auto it = t->pkIndex.find(key);
    if (it != t->pkIndex.end()) {
      hostCols(it->second.seg);
      if (!s.where || truthy(evalProg(henv, wp, it->second.row))) matched.push_back(it->second);
    }
  } else if ((ixm = findIndexEq(conj, *t, 0, key, ixFound))) {
    auto it = ixFound ? ixm->map.find(key) : ixm->map.end();
    if (it != ixm->map.end())
      for (const Loc& l : it->second) {
        hostCols(l.seg);
        if (!s.where || truthy(evalProg(henv, wp, l.row))) matched.push_back(l);
      }
  } else {
    comp.uploadSets();
    std::vector<int> parts = partsToScan(*t, 0, conj);
    std::vector<bool> inParts(t->nparts, false);
    for (int p : parts) inParts[p] = true;
    std::vector<std::pair<int32_t, DevVec<int32_t>>> sel;  // per segment: matching rows (on the GPU), how many
    std::vector<int64_t> nsel;
    for (size_t si = 0; si < t->segs.size(); ++si) {
      Segment& sg = *t->segs[si];
      if (!inParts[sg.part] || !segMayMatch(sg, conj)) continue;
      const uint32_t* sdel = sg.delDev();
      std::vector<void*> dvt;
      std::vector<uint8_t> evt;
      const BatchRef br{sg.part, sg.n, &sg.devs(dvt), sdel, &sg.encs(evt), 0, &sg};
      struct Temps { ~Temps() { releaseScanTemps(); } } temps;
      for (size_t k = 0; k < comp.slotCols.size(); ++k) bindView(comp.env.cols[k], br, comp.slotCols[k].second);
      scanFlush();
      DevVec<int32_t> idx(sg.n);
      const int64_t m = gpuSelect(comp.env, (bool)s.where, wp, sg.n, sdel, idx.p);
      if (m) sel.push_back({(int32_t)si, std::move(idx)}), nsel.push_back(m);
    }
    int64_t total = 0;
    for (int64_t m : nsel) total += m;
    matched.reserve((size_t)total);
    std::vector<int32_t> h;
    for (size_t k = 0; k < sel.size(); ++k) {
      h.resize((size_t)nsel[k]);
      d2h(h.data(), sel[k].second.p, sizeof(int32_t) * h.size());
      for (int32_t r : h) matched.push_back(Loc{sel[k].first, r});
    }
    for (int p : parts) {  // write-buffer rows: on the host, on all cores (chunks kept in row order)
      const WriteBuffer& b = *t->bufs[p];
      hostCols(bufLoc(p, 0).seg);
      const size_t nch = std::max<size_t>(1, std::min<size_t>(64, (size_t)b.n / 32768));
      std::vector<std::vector<Loc>> part(nch);
      parallelFor(nch, [&](size_t cb, size_t ce) {
        for (size_t ch = cb; ch < ce; ++ch)
          for (int64_t r = (int64_t)(b.n * ch / nch); r < (int64_t)(b.n * (ch + 1) / nch); ++r) {
            if (b.del->test(r)) continue;
            if (s.where && !truthy(evalProg(henv, wp, r))) continue;  // (NULL: not a match)
            part[ch].push_back(bufLoc(p, r));
          }
      }, 1);
      for (auto& v : part) matched.insert(matched.end(), v.begin(), v.end());
    }
  }

  if (rowsPending(*t, matched)) {  // row-level conflict with another open transaction: wait, then redo
    waitTxnFree(*t, lk, &matched);
    lk.unlock();
    return doModify(s);
  }
  // UPDATE = delete + re-insert of the modified row (like SingleStore's columnstore updates).
  if (s.type == Stmt::UPDATE && !undoLog()) {
    bool raw = true;
    for (auto& cm : t->cols) raw = raw && cm.pt != PType::VEC;
    std::set<int> reads;  // columns the SET expressions read
    std::function<bool(const ExprP&)> plain = [&](const ExprP& x) {
      if (!x) return true;
      if (x->kind == Expr::FUNC && x->op == "EMBED") return false;
      if (x->kind == Expr::COL) {
        if (x->bt != 0 || x->bc < 0) return false;
        reads.insert(x->bc);
      }
      for (auto& a : x->args)
        if (!plain(a)) return false;
      return true;
    };
    for (auto& [c, e] : sets) raw = raw && c != t->autoIncCol && plain(e);
    if (raw) {
      const auto u0 = Clock::now();
      ResultSet r = updateRaw(*t, matched, sets, reads);
      if (envSetting("TRACE")) fprintf(stderr, "[trace] update: match %.1f ms, rewrite %.1f ms (%zu rows)\n", std::chrono::duration<double, std::milli>(u0 - modT0).count(), msSince(u0), matched.size());
      return r;
    }
  }
  std::vector<std::vector<Value>> newRows;
  if (s.type == Stmt::UPDATE) {
    // EMBED(..) in SET: the texts of all matched rows go through the model in one batch first (the
    // per-row evaluation below then finds them in the cache).
    std::vector<ExprP> embeds;
    std::function<void(const ExprP&)> findEmbed = [&](const ExprP& x) {
      if (!x) return;
      if (x->kind == Expr::FUNC && x->op == "EMBED" && x->args.size() == 1) embeds.push_back(x->args[0]);
      for (auto& a : x->args) findEmbed(a);
    };
    for (auto& [c, e] : sets) findEmbed(e);
    if (!embeds.empty() && !matched.empty()) {
      std::vector<std::string> texts;
      for (const Loc& l : matched) {
        std::vector<Value> cur;
        for (size_t c = 0; c < t->cols.size(); ++c) cur.push_back(cellValue(*t, l, (int)c));
        PostCtx pc;
        pc.row = &cur;
        pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
        for (auto& a : embeds) {
          Value v = evalPost(a, pc);
          if (!v.isNull()) texts.push_back(v.toString());
        }
      }
      std::vector<std::string> ignored;
      embedCached(texts, ignored);
    }
    for (const Loc& l : matched) {  // SET expressions on the host, left to right (MySQL: a later one sees the new values)
      std::vector<Value> cur;
      for (size_t c = 0; c < t->cols.size(); ++c) cur.push_back(cellValue(*t, l, (int)c));
      PostCtx pc;
      pc.row = &cur;
      pc.colSlot = [](int bt, int bc) { return bt == 0 ? bc : -1; };
      std::vector<Value> row = cur;
      for (auto& [c, e] : sets) {
        Value v = evalPost(e, pc);
        if (!v.isNull() && t->cols[c].lt == LType::VARCHAR && v.kind != Value::STR) v = Value::S(v.toString());
        row[c] = v;
        cur[c] = v;
      }
      newRows.push_back(std::move(row));
    }
  }
  return applyModify(*t, matched, newRows);
}

// Deletes the matched rows (and inserts their new versions, for UPDATE). Holds the table lock.
ResultSet Engine::applyModify(Table& tt, const std::vector<Loc>& matched, const std::vector<std::vector<Value>>& newRows) {
  markDeleted(&tt, matched);
  for (auto& row : newRows) tt.appendRow(row);
  return okResult((int64_t)matched.size(), "Rows matched: " + std::to_string(matched.size()));
}

// Marks the matched rows deleted (undo records, PK and index entries, deleted-row bitmaps).
static void markDeleted(Table* t, const std::vector<Loc>& matched) {
  std::set<int> touched, touchedBufs;
  for (const Loc& l : matched) {
    if (undoLog()) {
      undoLog()->push_back({t, false, t->rowValues(l)});
      t->txnNoteDelete(l);
    }
    if (t->pk >= 0) t->pkIndex.erase(t->pkValue(l));
    if (!t->indexes.empty()) t->indexRemove(l);
    DelBits& d = delForWrite(l.seg < 0 ? t->bufs[bufPart(l)]->del : t->segs[l.seg]->del);  // (a private copy if a snapshot holds it)
    uint32_t& w = d.host[l.row >> 5];
    if (!(w & (1u << (l.row & 31)))) w |= 1u << (l.row & 31), ++d.ndel;
    if (l.seg < 0) ++t->bufs[bufPart(l)]->version, touchedBufs.insert(bufPart(l));
    else touched.insert(l.seg);
  }
  for (int si : touched) {  // bits changed: the device copy is re-uploaded on the next scan
    std::lock_guard<std::mutex> g(t->segs[si]->del->mu);
    ++t->segs[si]->del->ver;
  }
  for (int p : touchedBufs) {
    std::lock_guard<std::mutex> g(t->bufs[p]->del->mu);
    ++t->bufs[p]->del->ver;
  }
}

}  // namespace gdb
