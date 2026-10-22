#include "engine_core.h"

namespace gdb {
// ---- Snapshots (SNAPSHOT DATABASE): every database, table (schema + rows), view and pipeline in one
// file; the WAL then only holds what happened after it.
namespace {
void putU64(FILE* f, uint64_t v) { fwrite(&v, 8, 1, f); }
void putStr(FILE* f, std::string_view s) { putU64(f, s.size()); fwrite(s.data(), 1, s.size(), f); }
bool getU64(FILE* f, uint64_t& v) { return fread_unlocked(&v, 8, 1, f) == 1; }
bool getStr(FILE* f, std::string& s) {
  uint64_t n;
  if (!getU64(f, n) || n > (1ull << 32)) return false;
  s.resize(n);
  return fread_unlocked(s.data(), 1, n, f) == n;
}
void putVal(FILE* f, const Value& v) {
  uint8_t k = v.isNull() ? 0 : v.kind == Value::INT ? 1 : v.kind == Value::DBL ? 2 : v.kind == Value::STR ? 3 : v.kind == Value::DATE ? 4 : 5;
  fwrite(&k, 1, 1, f);
  if (k == 2) fwrite(&v.d, 8, 1, f);
  else if (k == 3) putStr(f, v.s);
  else if (k) fwrite(&v.i, 8, 1, f);
}
bool getVal(FILE* f, Value& v) {
  uint8_t k;
  if (fread(&k, 1, 1, f) != 1) return false;
  if (k == 0) { v = Value::Null(); return true; }
  if (k == 3) { std::string s; if (!getStr(f, s)) return false; v = Value::S(std::move(s)); return true; }
  int64_t i;
  double d;
  if (k == 2) { if (fread(&d, 8, 1, f) != 1) return false; v = Value::D(d); return true; }
  if (fread(&i, 8, 1, f) != 1) return false;
  v = k == 1 ? Value::I(i) : k == 4 ? Value::Dt(i) : Value::Ts(i);
  return true;
}
}  // namespace

ResultSet Engine::doSnapshot() {
  if (dir_.empty()) throw SqlError("SNAPSHOT needs a data directory (the server runs with --no-wal)", 1235);
  if (tTxn.active) commitTxn();
  const auto t0 = Clock::now();
  const int64_t loaded = loadedBytes_.load();
  writeSnapshot("", "");
  loadedBytes_ -= loaded;
  return okResult(0, "snapshot written in " + fmtMs(msSince(t0)) + " ms (writes paused " + fmtMs(lastSnapPauseMs_) + " ms)");
}

// Columnar table image: dictionaries, NULL flags, then per partition the live rows' physical column
// bytes (string columns as dictionary codes). Restoring it skips parsing and per-row inserts.
void putDbl(FILE* f, double v) {
  uint64_t u;
  memcpy(&u, &v, 8);
  putU64(f, u);
}
double getDbl(FILE* f) {
  uint64_t u = 0;
  getU64(f, u);
  double v;
  memcpy(&v, &u, 8);
  return v;
}

// ---- Dictionary files (snapshot format 5): <data dir>/dicts/<name>.wdict, append-only: "WDICT001",
// then frames of u32 "WDFR", u32 strings, u32 raw bytes, u32 compressed bytes, u32 CRC32 of the
// compressed bytes, and the zstd-compressed strings (u32 length + bytes each). A snapshot records
// (file, n, bytes): the first n strings end at byte `bytes`. Each snapshot appends only the strings
// added since the last one (before writes are paused); bytes after `bytes` (from a snapshot that
// was never installed) are cut off when the dictionary is loaded.
static constexpr uint32_t kDictFrame = 0x52464457;  // "WDFR"
static bool writeAt(int fd, const char* p, size_t len, uint64_t off) {
  while (len) {
    const ssize_t k = ::pwrite(fd, p, len, (off_t)off);
    if (k <= 0) return false;
    p += k, len -= (size_t)k, off += (uint64_t)k;
  }
  return true;
}
static void dictOrders(Dict& d, size_t upto, const std::string& dir);
static void dictPersist(Dict& d, size_t upto, const std::string& dir) {
  if (!d.fileName.empty() && upto <= d.fileStrs) {
    dictOrders(d, d.fileStrs, dir);
    return;
  }
  if (d.fileName.empty()) {
    mkdir(dir.c_str(), 0755);
    for (int tries = 0; d.fileName.empty(); ++tries) {
      char name[40];
      snprintf(name, sizeof name, "%016llx", (unsigned long long)(nowUs() * 6364136223846793005ull + d.uid * 1442695040888963407ull + tries));
      const int fd = ::open((dir + "/" + name + ".wdict").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
      if (fd < 0) {
        if (errno == EEXIST && tries < 8) continue;
        throw SqlError("cannot create a dictionary file in " + dir, 1105);
      }
      const bool ok = ::write(fd, "WDICT001", 8) == 8 && fdatasync(fd) == 0;
      ::close(fd);
      if (!ok) throw SqlError("cannot write a dictionary file in " + dir, 1105);
      d.fileName = name;
      d.fileStrs = 0;
      d.fileBytes = 8;
    }
  }
  std::vector<std::pair<size_t, size_t>> ranges;  // frames: up to 64K strings / 8 MB
  for (size_t i = d.fileStrs; i < upto;) {
    size_t j = i, bytes = 0;
    while (j < upto && j - i < 65536 && bytes < (8u << 20)) bytes += 4 + d.strs[j++].size();
    ranges.push_back({i, j});
    i = j;
  }
  std::vector<std::string> frames(ranges.size());
  std::atomic<bool> bad{false};
  ioParallel(ranges.size(), [&](size_t k) {
    std::string raw;
    for (size_t i = ranges[k].first; i < ranges[k].second; ++i) {
      const std::string_view x = d.strs[i];
      const uint32_t len = (uint32_t)x.size();
      raw.append((const char*)&len, 4).append(x);
    }
    std::string& fr = frames[k];
    fr.resize(20 + ZSTD_compressBound(raw.size()));
    const size_t z = ZSTD_compress(fr.data() + 20, fr.size() - 20, raw.data(), raw.size(), 3);
    if (ZSTD_isError(z)) {
      bad = true;
      return;
    }
    const uint32_t hdr[5] = {kDictFrame, (uint32_t)(ranges[k].second - ranges[k].first), (uint32_t)raw.size(), (uint32_t)z,
                             (uint32_t)crc32(0, (const Bytef*)fr.data() + 20, (uInt)z)};
    memcpy(fr.data(), hdr, 20);
    fr.resize(20 + z);
  });
  if (bad) throw SqlError("cannot compress a dictionary", 1105);
  const std::string path = dir + "/" + d.fileName + ".wdict";
  const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (fd < 0) throw SqlError("cannot open " + path, 1105);
  uint64_t off = d.fileBytes;
  bool ok = true;
  for (auto& fr : frames) {
    ok = ok && writeAt(fd, fr.data(), fr.size(), off);
    off += fr.size();
  }
  ok = ok && fdatasync(fd) == 0;
  ::close(fd);
  if (!ok) throw SqlError("cannot write " + path, 1105);
  d.fileBytes = off;
  d.fileStrs = upto;
  dictOrders(d, upto, dir);
}

// Side files with the collation orders (when cached for exactly the strings in the file), so a
// restart doesn't sort the dictionary again: "WORD0001", u64 strings, u32 CRC32C of the rest,
// zstd(order).
static void dictOrders(Dict& d, size_t upto, const std::string& dir) {
  for (int ci : {0, 1}) {
    if (d.orderFileStrs[ci] == upto) continue;
    auto order = sortedOrderCached(d, ci, upto);
    if (!order) continue;
    std::string z(ZSTD_compressBound(order->size() * 4), '\0');
    const size_t k = ZSTD_compress(z.data(), z.size(), order->data(), order->size() * 4, 1);
    if (ZSTD_isError(k)) continue;
    z.resize(k);
    std::string out = "WORD0001";
    const uint64_t nn = upto;
    const uint32_t crc = crc32c(z.data(), z.size());
    out.append((const char*)&nn, 8).append((const char*)&crc, 4).append(z);
    const std::string op = dir + "/" + d.fileName + ".o" + std::to_string(ci), tmp = op + ".tmp";
    const int ofd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (ofd < 0) continue;
    const bool wrote = writeAt(ofd, out.data(), out.size(), 0) && fdatasync(ofd) == 0;
    ::close(ofd);
    if (wrote && rename(tmp.c_str(), op.c_str()) == 0) d.orderFileStrs[ci] = upto;
    else unlink(tmp.c_str());
  }
}

static bool readFile(const std::string& path, std::string& out) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  struct stat st;
  bool ok = fstat(fd, &st) == 0;
  if (ok) {
    out.resize((size_t)st.st_size);
    size_t got = 0;  // (one read returns at most ~2 GB)
    while (ok && got < out.size()) {
      const ssize_t r = pread(fd, out.data() + got, out.size() - got, (off_t)got);
      if (r < 0 && errno == EINTR) continue;
      ok = r > 0;
      if (ok) got += (size_t)r;
    }
  }
  ::close(fd);
  return ok;
}

// The first n strings of a dictionary file (ending at byte `bytes`) into an empty dictionary.
static void dictLoad(Dict& d, const std::string& dir, const std::string& name, uint64_t n, uint64_t bytes) {
  const std::string path = dir + "/" + name + ".wdict";
  const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) throw SqlError("snapshot: dictionary file " + path + " is missing", 1105);
  struct Close {
    int fd;
    ~Close() { ::close(fd); }
  } closer{fd};
  std::string buf(bytes, '\0');
  size_t have = 0;  // (one read returns at most ~2 GB)
  for (ssize_t r; have < bytes && ((r = pread(fd, buf.data() + have, bytes - have, (off_t)have)) > 0 || (r < 0 && errno == EINTR));)
    if (r > 0) have += (size_t)r;
  if (bytes < 8 || have != bytes || memcmp(buf.data(), "WDICT001", 8) != 0)
    throw SqlError("snapshot: dictionary file " + path + " is truncated", 1105);
  struct Fr {
    size_t off, count, raw, z;
  };
  std::vector<Fr> frs;
  uint64_t got = 0;
  for (size_t off = 8; off < bytes;) {
    uint32_t hdr[5];
    if (off + 20 > bytes) throw SqlError("snapshot: dictionary file " + path + " is corrupt", 1105);
    memcpy(hdr, buf.data() + off, 20);
    if (hdr[0] != kDictFrame || off + 20 + hdr[3] > bytes || (uint32_t)crc32(0, (const Bytef*)buf.data() + off + 20, (uInt)hdr[3]) != hdr[4])
      throw SqlError("snapshot: dictionary file " + path + " is corrupt (frame at byte " + std::to_string(off) + ")", 1105);
    frs.push_back({off + 20, hdr[1], hdr[2], hdr[3]});
    got += hdr[1];
    off += 20 + hdr[3];
  }
  if (got != n) throw SqlError("snapshot: dictionary file " + path + " has " + std::to_string(got) + " strings, not " + std::to_string(n), 1105);
  std::vector<std::vector<std::string>> out(frs.size());
  std::atomic<bool> bad{false};
  ioParallel(frs.size(), [&](size_t k) {
    std::string raw(frs[k].raw, '\0');
    if (ZSTD_isError(ZSTD_decompress(raw.data(), raw.size(), buf.data() + frs[k].off, frs[k].z))) {
      bad = true;
      return;
    }
    auto& v = out[k];
    v.reserve(frs[k].count);
    for (size_t p = 0; p + 4 <= raw.size() && v.size() < frs[k].count;) {
      uint32_t len;
      memcpy(&len, raw.data() + p, 4);
      v.emplace_back(raw.data() + p + 4, len);
      p += 4 + len;
    }
    if (v.size() != frs[k].count) bad = true;
  });
  if (bad) throw SqlError("snapshot: dictionary file " + path + " is corrupt", 1105);
  for (auto& v : out)
    for (auto& x : v) d.appendUnique(std::move(x));
  struct stat st;
  if (fstat(fd, &st) == 0 && (uint64_t)st.st_size > bytes && ftruncate(fd, (off_t)bytes) != 0) {}  // (an uninstalled snapshot's tail)
  d.fileName = name;
  d.fileStrs = n;
  d.fileBytes = bytes;
  for (int ci : {0, 1}) {  // the collation orders: read when first needed (sortedDict)
    const std::string op = dir + "/" + name + ".o" + std::to_string(ci);
    if (access(op.c_str(), F_OK) == 0) d.orderFile[ci] = op;
  }
}

// A saved collation order (dictOrders) of exactly the dictionary's strings, or null.
std::shared_ptr<std::vector<int32_t>> loadOrderFile(const std::string& path, size_t n) {
  std::string o;
  if (!readFile(path, o) || o.size() < 20 || o.compare(0, 8, "WORD0001") != 0) return nullptr;
  uint64_t on;
  uint32_t crc;
  memcpy(&on, o.data() + 8, 8);
  memcpy(&crc, o.data() + 16, 4);
  if (on != n || crc32c(o.data() + 20, o.size() - 20) != crc) return nullptr;
  auto order = std::make_shared<std::vector<int32_t>>(n);
  const size_t k = ZSTD_decompress(order->data(), n * 4, o.data() + 20, o.size() - 20);
  if (ZSTD_isError(k) || k != n * 4) return nullptr;
  return order;
}

// After a snapshot is installed: dictionary files it doesn't refer to go.
static void dictFilesInstalled(const std::string& dir, const std::set<std::string>& used) {
  DIR* dp = opendir(dir.c_str());
  if (!dp) return;
  std::vector<std::string> gone;
  while (dirent* e = readdir(dp)) {
    std::string n = e->d_name;
    const size_t dot = n.find('.');
    if (dot != std::string::npos && dot == 16 && !used.count(n.substr(0, dot))) gone.push_back(n);  // (<name>.wdict / .o0 / .o1)
  }
  closedir(dp);
  for (auto& n : gone) unlink((dir + "/" + n).c_str());
}

// refs (snapshot.bin with a data dir): segments that are mostly live are not copied but referred to
// by their segment file (written now if they have none), with their deleted-row bitmap; their file
// ids are added to refs. Everything else is written row by row as before (and compacted).
// dictDir (format 5): dictionaries go to their files (dictPersist); their names are added to dictRefs.
void writeColumnar(FILE* f, const Table& t, std::vector<uint64_t>* refs = nullptr, const std::string* dictDir = nullptr,
                   std::set<std::string>* dictRefs = nullptr) {
  const size_t nc = t.cols.size();
  putU64(f, (uint64_t)t.nparts);
  for (size_t c = 0; c < nc; ++c) {
    Dict* d = t.dicts[c].get();
    const size_t n = d ? d->strs.size() : 0;
    putU64(f, n);
    if (dictDir) {
      if (n) {
        dictPersist(*d, n, *dictDir);  // (the strings added since the pre-pass)
        dictRefs->insert(d->fileName);
      }
      putStr(f, n ? d->fileName : std::string());
      putU64(f, n ? d->fileBytes : 0);
    } else if (d) {
      for (size_t i = 0; i < n; ++i) putStr(f, d->strs[i]);
    }
    putU64(f, t.hasNull[c]);
  }
  // The committed state: rows inserted by still-open transactions are left out, rows they deleted
  // are kept (their WAL records only appear at COMMIT, so recovery stays consistent).
  std::set<std::pair<int32_t, int32_t>> hidden, ghosts;
  for (auto& [k, v] : t.txnPending) {
    if (v.commitSeq || v.logged) continue;  // committed (or logged: its WAL record is rotated away with this snapshot)
    for (auto& l : v.hidden) hidden.insert({l.seg, l.row});
    for (auto& l : v.ghosts) ghosts.insert({l.seg, l.row});
  }
  auto keep = [&](int32_t seg, int64_t r, bool deleted) {
    if (hidden.empty() && ghosts.empty()) return !deleted;
    if (ghosts.count({seg, (int32_t)r})) return true;
    return !deleted && !hidden.count({seg, (int32_t)r});
  };
  std::vector<std::vector<int32_t>> live(t.segs.size());
  std::vector<bool> byRef(t.segs.size(), false);
  // (inside the snapshot's write pause: no per-row work for segments written by reference when no
  // transaction is open, their deleted-row bitmap is the answer)
  const bool plain = hidden.empty() && ghosts.empty();
  std::vector<int64_t> nlive(t.segs.size());
  for (size_t si = 0; si < t.segs.size(); ++si) {
    const auto& sg = t.segs[si];
    if (plain) {
      nlive[si] = sg->n - sg->del->ndel;
    } else {
      for (int64_t r = 0; r < sg->n; ++r)
        if (keep((int32_t)si, r, sg->del->test(r))) live[si].push_back((int32_t)r);
      nlive[si] = (int64_t)live[si].size();
    }
    byRef[si] = refs && tierEnabled() && sg->uses.size() == sg->dev.size() && sg->n > 0 && nlive[si] * 2 >= sg->n;
    if (plain && !byRef[si])
      for (int64_t r = 0; r < sg->n; ++r)
        if (!sg->del->test(r)) live[si].push_back((int32_t)r);
  }
  if (refs) {
    std::vector<size_t> rs;
    for (size_t si = 0; si < t.segs.size(); ++si)
      if (byRef[si]) rs.push_back(si);
    putU64(f, rs.size());
    for (size_t si : rs) {
      Segment& sg = *t.segs[si];
      auto file = segFileEnsure(*sg.d, sg.n, t.sortKey);
      segFileSync(*file);
      refs->push_back(file->id);
      putU64(f, (uint64_t)sg.part);
      putU64(f, (uint64_t)sg.n);
      putU64(f, file->id);
      for (size_t c = 0; c < nc; ++c) {
        putDbl(f, sg.mn[c]);
        putDbl(f, sg.mx[c]);
      }
      std::vector<uint32_t> bits((size_t)(sg.n + 31) / 32, ~0u);  // deleted unless live
      if (plain) {
        const auto& h = sg.del->host;
        for (size_t w = 0; w < bits.size(); ++w) bits[w] = w < h.size() ? h[w] : 0u;
      } else {
        for (int32_t r : live[si]) bits[r >> 5] &= ~(1u << (r & 31));
      }
      if (sg.n % 32) bits.back() &= (1u << (sg.n % 32)) - 1;
      fwrite(bits.data(), 4, bits.size(), f);
    }
  }
  std::vector<uint8_t> tmp;
  for (int p = 0; p < t.nparts; ++p) {
    std::vector<std::pair<const Segment*, const std::vector<int32_t>*>> parts;  // live rows per segment
    int64_t n = 0;
    for (size_t si = 0; si < t.segs.size(); ++si) {
      if (t.segs[si]->part != p || byRef[si]) continue;
      n += (int64_t)live[si].size();
      parts.push_back({t.segs[si].get(), &live[si]});
    }
    const WriteBuffer& b = *t.bufs[p];
    std::vector<int32_t> blive;
    for (int64_t r = 0; r < b.n; ++r)
      if (keep(-1 - p, r, b.del->test(r))) blive.push_back((int32_t)r);
    n += (int64_t)blive.size();
    putU64(f, (uint64_t)n);
    std::vector<uint8_t> out;
    for (size_t c = 0; c < nc; ++c) {
      const int sz = t.colBytes((int)c);
      out.resize((size_t)n * sz);
      size_t o = 0;
      auto copy = [&](const uint8_t* src, const std::vector<int32_t>& rows, int64_t total) {
        if ((int64_t)rows.size() == total) {
          memcpy(out.data() + o, src, (size_t)total * sz);
          o += (size_t)total * sz;
          return;
        }
        for (int32_t r : rows) memcpy(out.data() + o, src + (size_t)r * sz, sz), o += sz;
      };
      for (auto& [sg, lv] : parts) copy(segRawColumn(*sg->d, (int)c, tmp), *lv, sg->n);
      copy(b.cols[c].data(), blive, b.n);
      fwrite(out.data(), 1, out.size(), f);
    }
  }
}

void readColumnar(FILE* f, Table& t, bool refs = false, const std::string* dictDir = nullptr) {
  const size_t nc = t.cols.size();
  uint64_t np, v;
  getU64(f, np);
  for (size_t c = 0; c < nc; ++c) {
    getU64(f, v);
    // (codes are positions: appended as they are, indexed when first looked up)
    if (v && (!t.dicts[c] || t.dicts[c]->strs.size() != 0)) throw SqlError("snapshot: unexpected dictionary for " + t.cols[c].name, 1105);
    if (dictDir) {  // format 5: in its dictionary file
      std::string name;
      uint64_t bytes = 0;
      getStr(f, name);
      getU64(f, bytes);
      if (v) dictLoad(*t.dicts[c], *dictDir, name, v, bytes);
    } else {
      for (uint64_t i = 0; i < v; ++i) {
        std::string x;
        getStr(f, x);
        t.dicts[c]->appendUnique(std::move(x));
      }
    }
    getU64(f, v);
    t.hasNull[c] = (uint8_t)v;
  }
  const bool samePlacement = (int)np == t.nparts || t.nparts == 1;
  auto place = [&](const std::vector<std::vector<uint8_t>>& colsIn, int dst, const std::vector<int64_t>* rows, size_t cnt) {
    WriteBuffer& b = *t.bufs[dst];
    for (size_t c = 0; c < nc; ++c) {
      const int sz = t.colBytes((int)c);
      auto& bc = b.cols[c];
      if (!rows) bc.insert(bc.end(), colsIn[c].begin(), colsIn[c].begin() + cnt * sz);
      else
        for (int64_t r : *rows) bc.insert(bc.end(), colsIn[c].begin() + r * sz, colsIn[c].begin() + (r + 1) * sz);
    }
    b.n += (int64_t)(rows ? rows->size() : cnt);
    b.del->host.resize((b.n + 31) / 32, 0);
    ++b.version;
  };
  // Rows of partition p of the snapshot (or of any, when keep is given: only those rows) -> write buffers.
  auto ingest = [&](const std::vector<std::vector<uint8_t>>& colsIn, uint64_t p, uint64_t n, const std::vector<int64_t>* keepRows) {
    if (samePlacement) {
      place(colsIn, t.nparts == 1 ? 0 : (int)p, keepRows, n);
      return;
    }
    // a different partition count: re-hash each row by its shard key
    std::vector<std::vector<int64_t>> to(t.nparts);
    std::vector<int64_t> ints(nc);
    std::vector<double> dbls(nc);
    auto route = [&](int64_t r) {
      for (int c : t.shardKey) {
        const uint8_t* x = colsIn[c].data() + r * t.colBytes(c);
        if (t.cols[c].pt == PType::F64) memcpy(&dbls[c], x, 8);
        else if (t.cols[c].pt == PType::I64) memcpy(&ints[c], x, 8);
        else { int32_t i32; memcpy(&i32, x, 4); ints[c] = i32; }
      }
      to[t.partOf(ints, dbls)].push_back(r);
    };
    if (keepRows)
      for (int64_t r : *keepRows) route(r);
    else
      for (uint64_t r = 0; r < n; ++r) route((int64_t)r);
    for (int q = 0; q < t.nparts; ++q) place(colsIn, q, &to[q], 0);
  };
  if (refs) {  // segments kept in their files: back as cold segments (columns read when used)
    uint64_t nref = 0;
    getU64(f, nref);
    for (uint64_t i = 0; i < nref; ++i) {
      uint64_t part, n, id;
      getU64(f, part);
      getU64(f, n);
      getU64(f, id);
      auto seg = std::make_unique<Segment>();
      seg->id = nextSegmentId();
      seg->part = t.nparts == 1 ? 0 : (int)part;
      seg->n = (int64_t)n;
      seg->host.resize(nc);
      seg->dev.assign(nc, nullptr);
      seg->mapped.assign(nc, 0);
      seg->uses.assign(nc, 0);
      seg->heat.assign(nc, 0.f);
      seg->mn.resize(nc);
      seg->mx.resize(nc);
      seg->encHost.resize(nc);
      for (size_t c = 0; c < nc; ++c) {
        seg->mn[c] = getDbl(f);
        seg->mx[c] = getDbl(f);
      }
      std::vector<uint32_t> bits((size_t)(n + 31) / 32);
      if (fread(bits.data(), 4, bits.size(), f) != bits.size()) throw SqlError("truncated snapshot", 1105);
      seg->d->file = segFileOpen(id, nc, (int64_t)n);
      seg->enc.resize(nc);
      for (size_t c = 0; c < nc; ++c) {
        seg->enc[c] = seg->d->file->cols[c].enc;
        if (seg->d->file->cols[c].width != (uint32_t)t.colBytes((int)c)) throw SqlError("segment file does not match the table", 1105);
      }
      segAccount(*seg->d);
      seg->d->cold.assign(nc, 1);
      int64_t ndel = 0;
      for (uint32_t w : bits) ndel += __builtin_popcount(w);
      if (!samePlacement) {  // rows go to their new partitions through the write buffers
        std::vector<std::vector<uint8_t>> colsIn(nc);
        std::vector<uint8_t> tmp;
        for (size_t c = 0; c < nc; ++c) {
          const uint8_t* raw = segRawColumn(*seg->d, (int)c, tmp);
          colsIn[c].assign(raw, raw + n * t.colBytes((int)c));
        }
        std::vector<int64_t> keepRows;
        for (uint64_t r = 0; r < n; ++r)
          if (!((bits[r >> 5] >> (r & 31)) & 1u)) keepRows.push_back((int64_t)r);
        ingest(colsIn, part, n, &keepRows);
        continue;
      }
      seg->del->host = std::move(bits);
      seg->del->ndel = ndel;
      const int32_t segId = (int32_t)t.segs.size();
      if (t.pk >= 0) {
        std::vector<uint8_t> tmp;
        const uint8_t* raw = segRawColumn(*seg->d, t.pk, tmp);
        const PType pt = t.cols[t.pk].pt;
        for (int64_t r = 0; r < (int64_t)n; ++r) {
          if (seg->del->test(r)) continue;
          const int64_t k = pt == PType::I32 ? ((const int32_t*)raw)[r] : pt == PType::I64 ? ((const int64_t*)raw)[r] : (int64_t)((const double*)raw)[r];
          t.pkIndex[k] = Loc{segId, (int32_t)r};
        }
      }
      t.segs.push_back(std::move(seg));
    }
  }
  for (uint64_t p = 0; p < np; ++p) {
    uint64_t n;
    getU64(f, n);
    std::vector<std::vector<uint8_t>> colsIn(nc);
    for (size_t c = 0; c < nc; ++c) {
      colsIn[c].resize(n * t.colBytes((int)c));
      if (fread(colsIn[c].data(), 1, colsIn[c].size(), f) != colsIn[c].size()) throw SqlError("truncated snapshot", 1105);
    }
    ingest(colsIn, p, n, nullptr);
    t.flush(true);  // segments (sorted by the SORT KEY, compressed, primary-key index) right away
  }
  for (auto& ix : t.indexes) t.buildIndex(ix);
}

// Writes a snapshot of everything (path "" = the data dir's snapshot.bin, which also truncates the
// WAL) or of one database (BACKUP DATABASE).
// zstd-compressed snapshot / backup streams: a FILE* (fopencookie) the snapshot code writes and
// reads as usual.
namespace {
struct ZW {
  FILE* raw;
  ZSTD_CCtx* c;
  std::vector<char> out;
};
ssize_t zwWrite(void* ck, const char* buf, size_t n) {
  auto* z = (ZW*)ck;
  ZSTD_inBuffer ib{buf, n, 0};
  while (ib.pos < ib.size) {
    ZSTD_outBuffer ob{z->out.data(), z->out.size(), 0};
    size_t rc = ZSTD_compressStream2(z->c, &ob, &ib, ZSTD_e_continue);
    if (ZSTD_isError(rc)) return -1;
    if (fwrite(z->out.data(), 1, ob.pos, z->raw) != ob.pos) return -1;
  }
  return (ssize_t)n;
}
int zwClose(void* ck) {
  auto* z = (ZW*)ck;
  ZSTD_inBuffer ib{nullptr, 0, 0};
  size_t rc;
  do {
    ZSTD_outBuffer ob{z->out.data(), z->out.size(), 0};
    rc = ZSTD_compressStream2(z->c, &ob, &ib, ZSTD_e_end);
    fwrite(z->out.data(), 1, ob.pos, z->raw);
  } while (rc > 0 && !ZSTD_isError(rc));
  fflush(z->raw);
  fdatasync(fileno(z->raw));
  int r = fclose(z->raw);
  ZSTD_freeCCtx(z->c);
  delete z;
  return r;
}
FILE* zstdWriter(FILE* raw) {
  auto* z = new ZW{raw, ZSTD_createCCtx(), std::vector<char>(ZSTD_CStreamOutSize())};
  ZSTD_CCtx_setParameter(z->c, ZSTD_c_compressionLevel, 3);
  ZSTD_CCtx_setParameter(z->c, ZSTD_c_nbWorkers, 4);
  ZSTD_CCtx_setParameter(z->c, ZSTD_c_checksumFlag, 1);  // (a damaged snapshot / backup fails to load instead of loading garbage)
  cookie_io_functions_t io{nullptr, zwWrite, nullptr, zwClose};
  FILE* f = fopencookie(z, "w", io);
  if (f) setvbuf(f, nullptr, _IOFBF, 1 << 20);
  return f;
}
struct ZR {
  FILE* raw;
  ZSTD_DCtx* d;
  std::vector<char> in;
  ZSTD_inBuffer ib{nullptr, 0, 0};
};
ssize_t zrRead(void* ck, char* buf, size_t n) {
  auto* z = (ZR*)ck;
  ZSTD_outBuffer ob{buf, n, 0};
  while (ob.pos == 0) {
    if (z->ib.pos == z->ib.size) {
      size_t got = fread(z->in.data(), 1, z->in.size(), z->raw);
      if (got == 0) break;
      z->ib = {z->in.data(), got, 0};
    }
    size_t rc = ZSTD_decompressStream(z->d, &ob, &z->ib);
    if (ZSTD_isError(rc)) return -1;
  }
  return (ssize_t)ob.pos;
}
int zrClose(void* ck) {
  auto* z = (ZR*)ck;
  int r = fclose(z->raw);
  ZSTD_freeDCtx(z->d);
  delete z;
  return r;
}
FILE* zstdReader(FILE* raw) {
  auto* z = new ZR{raw, ZSTD_createDCtx(), std::vector<char>(ZSTD_DStreamInSize())};
  cookie_io_functions_t io{zrRead, nullptr, nullptr, zrClose};
  FILE* f = fopencookie(z, "r", io);
  if (f) setvbuf(f, nullptr, _IOFBF, 1 << 20);  // (millions of small reads: dictionary strings)
  return f;
}
}  // namespace

void Engine::writeSnapshot(const std::string& outPath, const std::string& onlyDb, uint64_t* lsnOut, std::pair<uint64_t, int64_t>* posOut) {
  std::vector<std::shared_ptr<Table>> tabs;
  std::set<std::string> dbs;
  std::map<std::string, std::string> views;
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& [k, t] : cat_.tables)
      if (t->db != "__tmp" && (onlyDb.empty() || lower(t->db) == lower(onlyDb))) tabs.push_back(t);
    for (auto& d : cat_.dbs)
      if (onlyDb.empty() || d == lower(onlyDb)) dbs.insert(d);
    for (auto& [k, v] : cat_.views)
      if (onlyDb.empty() || k.rfind(lower(onlyDb) + ".", 0) == 0) views[k] = v;
  }
  std::lock_guard<std::mutex> snapGuard(snapMu_);
  const bool v4 = outPath.empty() && tierEnabled();  // snapshot.bin: segments by reference (their files)
  std::vector<uint64_t> refs;
  const std::string dictDir = dir_ + "/dicts";
  std::set<std::string> dictRefs;
  // The pre-pass runs twice: what a concurrent load added during the first one (segments, strings)
  // is written by the second, so little is left for the pause.
  for (int pass = 0; pass < (v4 ? 2 : 0); ++pass) {
  if (v4)  // big write buffers become segments first, so they too are written outside the pause below
    for (auto& t : tabs) {
      std::unique_lock<FairSharedMutex> lk(t->mu);
      for (int p = 0; p < t->nparts; ++p)
        if (t->bufs[p]->n - t->bufs[p]->ndel() >= 65536) t->flushPart(p);
    }
  if (v4)  // write the files first, without quiescing anything (segment data is immutable)
    for (auto& t : tabs) {
      std::vector<std::shared_ptr<Segment>> segs;
      {
        std::shared_lock<FairSharedMutex> lk(t->mu);
        for (auto& sp : t->segs)
          if (sp->uses.size() == sp->dev.size() && sp->n > 0 && sp->ndel() * 2 <= sp->n) segs.push_back(sp);
      }
      for (auto& sp : segs) segFileSync(*segFileEnsure(*sp->d, sp->n, t->sortKey));
    }
  if (v4)  // dictionaries: the strings added since the last snapshot go to their files (no pause)
    for (auto& t : tabs) {
      std::vector<std::shared_ptr<Dict>> ds;
      {
        std::shared_lock<FairSharedMutex> lk(t->mu);
        ds = t->dicts;
      }
      for (auto& d : ds)
        if (d && d->strs.size()) dictPersist(*d, d->strs.size(), dictDir);
    }
  }
  const auto pause0 = Clock::now();
  std::unique_lock<FairSharedMutex> gate(writeGate_, std::defer_lock);  // (statements in flight log first)
  lastSnapSkipped_ = false;
  if (tSnapTryOnly && loadsInFlight_ > 0) {  // (automatic: not behind a long LOAD, writers would queue behind it)
    lastSnapSkipped_ = true;
    return;
  }
  if (tWriteGateHeld) throw SqlError("internal: a snapshot inside a statement that logs to the WAL (would wait for itself)", 1105);
  gate.lock();  // (fair: statements in flight finish, new ones wait for the snapshot)
  std::vector<std::unique_lock<FairSharedMutex>> locks;  // quiesce every table
  std::sort(tabs.begin(), tabs.end());
  for (auto& t : tabs) locks.emplace_back(t->mu);
  std::lock_guard<std::mutex> wg(walMu_);
  if (lsnOut) *lsnOut = lsn_;  // the snapshot contains exactly the records up to this lsn
  if (posOut) *posOut = {walGen_, walOff_};  // ... and exactly the WAL up to here
  const std::string fin = outPath.empty() ? dir_ + "/snapshot.bin" : outPath, tmp = fin + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) throw SqlError("cannot write " + tmp, 1105);
  // GDBSNAP2+: the rest is one zstd stream; 3: columnar tables; 4: + segment files; 5: + dictionary files
  fwrite(v4 ? "GDBSNAP5" : "GDBSNAP3", 1, 8, f);
  f = zstdWriter(f);
  putU64(f, dbs.size());
  for (auto& d : dbs) putStr(f, cat_.syncDbs.count(d) ? d + "\x1fSYNC" : d);  // (cat_.mu not needed: dbs only grow under writes, quiesced here)
  putU64(f, views.size());
  for (auto& [k, v] : views) { putStr(f, k); putStr(f, v); }
  int64_t rows = 0;
  putU64(f, tabs.size());
  for (auto& t : tabs) {
    putStr(f, t->db);
    putStr(f, showCreate(*t));
    putU64(f, (uint64_t)t->autoNext);
    putU64(f, (uint64_t)t->liveRows());
    writeColumnar(f, *t, v4 ? &refs : nullptr, v4 ? &dictDir : nullptr, &dictRefs);
    rows += t->liveRows();
  }
  std::vector<std::shared_ptr<Pipeline>> ps;
  {
    std::lock_guard<std::mutex> g(pipesMu_);
    for (auto& [k, p] : pipes_)
      if (onlyDb.empty() || lower(p->db) == lower(onlyDb)) ps.push_back(p);
  }
  putU64(f, ps.size());
  for (auto& p : ps) {
    std::lock_guard<std::mutex> g(p->mu);
    for (auto* x : {&p->name, &p->db, &p->table, &p->glob, &p->delim, &p->cols}) putStr(f, *x);
    putU64(f, (uint64_t)p->skip);
    putU64(f, p->wantRun ? 1 : 0);
    putU64(f, p->done.size());
    for (auto& d : p->done) putStr(f, d);
  }
  fflush(f);
  fdatasync(fileno(f));
  fclose(f);
  auto syncDir = [&](const std::string& file) {
    std::string d = file.substr(0, file.rfind('/') == std::string::npos ? 0 : file.rfind('/'));
    int fd = open(d.empty() ? "." : d.c_str(), O_RDONLY);
    if (fd >= 0) { fsync(fd); ::close(fd); }
  };
  {
    std::lock_guard<std::mutex> ug(usersMu_);  // users go into a side file next to the snapshot (tmp + rename)
    const std::string uf = fin + ".users", ut = uf + ".tmp";
    if (FILE* u = fopen(ut.c_str(), "wb")) {
      putU64(u, users_.size());
      for (auto& [name, h] : users_) { putStr(u, name); putStr(u, h); }
      auto defs = routineDefs();
      putU64(u, defs.size());
      for (auto& d : defs) putStr(u, d);
      putU64(u, grants_.size());
      for (auto& [name, es] : grants_) {
        putStr(u, name);
        putU64(u, es.size());
        for (auto& e : es) putStr(u, e);
      }
      fflush(u);
      fdatasync(fileno(u));
      fclose(u);
      rename(ut.c_str(), uf.c_str());
    }
  }
  if (outPath.empty()) {
    // Crash-safe order: (1) the WAL so far moves to wal.old and a new WAL starts, (2) the snapshot is
    // installed, (3) wal.old is deleted. Recovery replays wal.old only if snapshot.bin.tmp is still
    // there (the snapshot was not installed), so every change is applied exactly once.
    if (wal_) fclose(wal_);
    wal_ = nullptr;
    rename((dir_ + "/wal.log").c_str(), (dir_ + "/wal.old").c_str());
    ++walGen_;
    walOpen(true);
    if (wal_) fdatasync(fileno(wal_));
    syncDir(fin);
    segDirSync();
  }
  if (rename(tmp.c_str(), fin.c_str()) != 0) throw SqlError("cannot install snapshot " + fin, 1105);
  syncDir(fin);
  if (v4) segFilesInstalled(refs), dictFilesInstalled(dictDir, dictRefs);  // files only the previous snapshot used can go
  if (outPath.empty()) {
    archiveWal(dir_ + "/wal.old");
    archiveLoads(walGen_);
  }
  lastSnapPauseMs_ = msSince(pause0);
  (void)rows;
}

std::string gWalArchive;
int64_t gSnapshotTriggerBytes = (int64_t)256 << 20;  // (replay runs ~150K records/s: a crash costs ~20 s at most)

uint32_t fileCrc(const std::string& path) {  // CRC32C of a file's bytes (0xFFFFFFFF: unreadable)
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return 0xFFFFFFFFu;
  std::vector<char> buf(8 << 20);
  uint32_t c = 0;
  for (ssize_t k; (k = ::read(fd, buf.data(), buf.size())) > 0;) c = crc32c(buf.data(), (size_t)k, c);
  ::close(fd);
  return c;
}
// Copies a file, checks the copy against the source (a damaged copy would replay to other rows) and
// returns its CRC32C; one more attempt on a mismatch.
void copyFileOnce(const std::string& from, const std::string& to);
uint32_t copyFile(const std::string& from, const std::string& to) {
  const uint32_t want = fileCrc(from);
  for (int attempt = 0;; ++attempt) {
    copyFileOnce(from, to);
    const uint32_t got = fileCrc(to);
    if (got == want) return got;
    fprintf(stderr, "[warpstore] copy of %s differs from the file (CRC %08x, want %08x)%s\n", from.c_str(), got, want, attempt ? "" : ": copying again");
    if (attempt) {
      unlink(to.c_str());
      throw SqlError("the copy of " + from + " kept differing from the file (memory or disk errors?)", 1105);
    }
  }
}
void copyFileOnce(const std::string& from, const std::string& to) {
  int in = open(from.c_str(), O_RDONLY);
  if (in < 0) throw SqlError("cannot read " + from, 1105);
  int out = open((to + ".tmp").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (out < 0) {
    ::close(in);
    throw SqlError("cannot write " + to, 1105);
  }
  struct stat st;
  fstat(in, &st);
  off_t left = st.st_size;
  bool ok = true;
  while (left > 0) {  // in-kernel copy (a reflink on filesystems that have them)
    ssize_t k = copy_file_range(in, nullptr, out, nullptr, (size_t)std::min<off_t>(left, 1 << 30), 0);
    if (k <= 0) {
      ok = false;
      break;
    }
    left -= k;
  }
  if (!ok) {  // e.g. not supported across these filesystems: plain copy
    lseek(in, 0, SEEK_SET);
    lseek(out, 0, SEEK_SET);
    if (ftruncate(out, 0) != 0) {}
    std::vector<char> buf(1 << 20);
    ok = true;
    for (ssize_t k; ok && (k = read(in, buf.data(), buf.size())) > 0;) ok = write(out, buf.data(), (size_t)k) == k;
  }
  ok = ok && fdatasync(out) == 0;
  ::close(in);
  ::close(out);
  if (!ok || rename((to + ".tmp").c_str(), to.c_str()) != 0) {
    unlink((to + ".tmp").c_str());
    throw SqlError("cannot copy " + from + " to " + to, 1105);
  }
}

void mkdirs(const std::string& d) {
  for (size_t p = 1; p <= d.size(); ++p)
    if (p == d.size() || d[p] == '/') mkdir(d.substr(0, p).c_str(), 0755);
}

static uint64_t walFileGen(const std::string& path) {
  uint64_t g = 0;
  if (FILE* f = fopen(path.c_str(), "rb")) {
    char hdr[16];
    if (fread(hdr, 1, 16, f) == 16 && memcmp(hdr, "GDBWAL3\n", 8) == 0) memcpy(&g, hdr + 8, 8);
    fclose(f);
  }
  return g;
}

void Engine::archiveWal(const std::string& path) {
  if (access(path.c_str(), F_OK) != 0) return;
  if (!gWalArchive.empty()) {
    const uint64_t g = walFileGen(path);
    if (g) {
      try {
        mkdirs(gWalArchive);
        char name[64];
        snprintf(name, sizeof name, "/wal-%020llu.log", (unsigned long long)g);
        const std::string to = gWalArchive + name;
        if (rename(path.c_str(), to.c_str()) != 0) copyFile(path, to);
      } catch (std::exception& e) {
        fprintf(stderr, "[warpstore] WAL archive: %s (keeping %s)\n", e.what(), path.c_str());
        return;
      }
    }
  }
  unlink(path.c_str());
}

// Server-side LOAD DATA INFILE copies its file to <data dir>/loads/g<gen>-.. before logging it, so the
// WAL never depends on files outside the data directory. Once a snapshot covers a WAL generation its
// copies go to the archive (they are needed to replay archived WALs) or are deleted.
void Engine::archiveLoads(uint64_t belowGen) {
  const std::string d = dir_ + "/loads";
  DIR* dp = opendir(d.c_str());
  if (!dp) return;
  std::vector<std::string> names;
  while (dirent* e = readdir(dp))
    if (e->d_name[0] == 'g') names.push_back(e->d_name);
  closedir(dp);
  for (auto& n : names) {
    if (strtoull(n.c_str() + 1, nullptr, 10) >= belowGen) continue;
    const std::string from = d + "/" + n;
    if (!gWalArchive.empty()) {
      try {
        mkdirs(gWalArchive + "/loads");
        const std::string to = gWalArchive + "/loads/" + n;
        if (rename(from.c_str(), to.c_str()) != 0) copyFile(from, to);
      } catch (std::exception& e) {
        fprintf(stderr, "[warpstore] WAL archive: %s\n", e.what());
        continue;
      }
    }
    unlink(from.c_str());
  }
}

// ---- Users: mysql_native_password (token = SHA1(pw) XOR SHA1(scramble + SHA1(SHA1(pw)))).
static std::string sha256(const std::string& x) {
  unsigned char d[32];
  SHA256((const unsigned char*)x.data(), x.size(), d);
  return std::string((const char*)d, 32);
}
static std::string sha1(const std::string& x) {
  unsigned char d[20];
  SHA1((const unsigned char*)x.data(), x.size(), d);
  return std::string((const char*)d, 20);
}

int Engine::checkAuthSha2(const std::string& user, const std::string& scramble, const std::string& token) {
  std::lock_guard<std::mutex> g(usersMu_);
  auto it = users_.find(user.empty() ? "root" : user);
  if (it == users_.end() || it->second == "!role") return 0;
  if (it->second.empty()) return token.empty() || (token.size() == 1 && token[0] == 0) ? 1 : 0;  // no password
  if (it->second.size() != 52) return -1;  // created before SHA256 hashes were kept
  const std::string stage2 = it->second.substr(20);
  if (token.size() != 32) return 0;
  const std::string x = sha256(stage2 + scramble);
  std::string stage1(32, '\0');
  for (int i = 0; i < 32; ++i) stage1[i] = (char)(token[i] ^ x[i]);
  return sha256(stage1) == stage2 ? 1 : 0;
}

bool Engine::checkAuth(const std::string& user, const std::string& scramble, const std::string& token) {
  std::lock_guard<std::mutex> g(usersMu_);
  auto it = users_.find(user.empty() ? "root" : user);
  if (it == users_.end()) return false;
  const std::string stage2 = it->second.substr(0, std::min<size_t>(20, it->second.size()));
  if (stage2.empty()) return user == "root" || user.empty() ? true : token.empty();  // root without a password: open
  if (token.size() != 20) return false;
  std::string h = sha1(scramble + stage2), stage1(20, '\0');
  for (int i = 0; i < 20; ++i) stage1[i] = (char)(token[i] ^ h[i]);
  return sha1(stage1) == stage2;
}

thread_local std::string tUser = "root";
thread_local bool tLocalInfile = false;
thread_local bool tDescribeOnly = false;
const std::string& sessionUser() { return tUser; }
const std::string& tUserName() { return tUser; }

bool Engine::checkPassword(const std::string& user, const std::string& pw) {
  std::lock_guard<std::mutex> g(usersMu_);
  auto it = users_.find(user.empty() ? "root" : user);
  if (it == users_.end() || it->second == "!role") return false;
  return it->second.empty() ? pw.empty() : it->second.substr(0, 20) == sha1(sha1(pw));
}

bool Engine::isSuper(const std::string& user) {
  if (user == "root") return true;
  std::lock_guard<std::mutex> g(usersMu_);
  return grants_[user].count("ALL|*.*") > 0;
}
void Engine::setUser(const std::string& user) { tUser = user.empty() ? "root" : user; }

bool Engine::hasPriv(const std::string& user, const std::string& priv, const std::string& db, const std::string& table) {
  std::set<std::string> seen;
  std::function<bool(const std::string&)> rec = [&](const std::string& u) {
    if (u == "root") return true;
    if (!seen.insert(u).second) return false;
    auto it = grants_.find(u);
    if (it == grants_.end()) return false;
    for (const std::string& e : it->second) {
      if (e[0] == '@') {
        if (rec(e.substr(1))) return true;
        continue;
      }
      const std::string pv = e.substr(0, e.find('|')), sc = e.substr(e.find('|') + 1);
      if (pv != "ALL" && pv != priv) continue;
      if (sc == "*.*" || sc == db + ".*" || sc == db + "." + table || (sc.find('.') == std::string::npos && sc == table)) return true;
    }
    return false;
  };
  return rec(user);
}

// Statement-level privilege checks (SELECT / INSERT / UPDATE / DELETE / CREATE / DROP / ALTER per
// table or database; account management, pipelines, backups need ALL ON *.*). Views are checked by
// their own name, so a view over a filtered table is how row-level security is expressed.
void Engine::checkPrivileges(const Stmt& s) {
  if (tUser == "root") return;
  std::lock_guard<std::mutex> g(usersMu_);
  if (hasPriv(tUser, "ALL", "*", "*") && grants_[tUser].count("ALL|*.*")) return;
  auto need = [&](const std::string& priv, const std::string& name) {
    if (name.empty()) return;
    std::string db = lower(dbOf(name)), t = lower(plainName(name));
    if (db == "information_schema" || db == "__tmp") return;
    if (!hasPriv(tUser, priv, db, t) && !hasPriv(tUser, priv, db, "*"))
      throw SqlError(priv + " command denied to user '" + tUser + "'@'%' for table '" + t + "'", 1142);
  };
  auto needDb = [&](const std::string& priv, const std::string& db) {
    if (!hasPriv(tUser, priv, lower(db), "*") && !grants_[tUser].count(priv + "|" + lower(db) + ".*") &&
        !hasPriv(tUser, priv, lower(db), ""))
      throw SqlError("Access denied for user '" + tUser + "'@'%' to database '" + db + "'", 1044);
  };
  auto super = [&] {
    if (!grants_[tUser].count("ALL|*.*") && !hasPriv(tUser, "SUPER", "", ""))
      throw SqlError("Access denied; you need (at least one of) the SUPER privilege(s) for this operation", 1227);
  };
  std::function<void(const Stmt&, std::set<std::string>)> reads = [&](const Stmt& q, std::set<std::string> ctes) {
    for (auto& [n, c] : q.ctes) {
      ctes.insert(lower(n));
      reads(*c, ctes);
    }
    std::function<void(const ExprP&)> ex = [&](const ExprP& e) {
      if (!e) return;
      if (e->sub) reads(*e->sub, ctes);
      for (auto& a : e->args) ex(a);
    };
    for (auto& f : q.from) {
      if (f.sub) reads(*f.sub, ctes);
      else if (!ctes.count(lower(f.name))) need("SELECT", f.name);
      ex(f.on);
    }
    for (auto& it : q.items) ex(it.e);
    ex(q.where);
    ex(q.having);
    for (auto& c : q.joinConds) ex(c);
    if (q.unionNext) reads(*q.unionNext, ctes);
    if (q.select) reads(*q.select, ctes);
  };
  switch (s.type) {
    case Stmt::SELECT:
      reads(s, {});
      if (!s.outfile.empty()) super();  // SELECT .. INTO OUTFILE writes server files: FILE privilege
      break;
    case Stmt::EXPLAIN: case Stmt::PROFILE: if (s.inner) reads(*s.inner, {}); break;
    case Stmt::INSERT: need("INSERT", s.table); reads(s, {}); break;
    case Stmt::LOAD:  // reading server-side files / remote sources needs FILE (SUPER here), as in MySQL
      need("INSERT", s.table);
      if (!tLocalInfile) super();
      break;
    case Stmt::UPDATE: need("UPDATE", s.table); reads(s, {}); break;
    case Stmt::DELETE: need("DELETE", s.table); reads(s, {}); break;
    case Stmt::CREATE: if (!s.temporary) need("CREATE", s.table); if (s.select) reads(*s.select, {}); break;
    case Stmt::CREATE_VIEW: need("CREATE VIEW", s.table); break;
    case Stmt::DROP: for (auto& n : s.names) need("DROP", n); break;
    case Stmt::DROP_VIEW: for (auto& n : s.names) need("DROP", n); break;
    case Stmt::TRUNCATE: need("DROP", s.table); break;
    case Stmt::ALTER: need("ALTER", s.table); break;
    case Stmt::RENAME: for (auto& n : s.names) need("ALTER", n); break;
    case Stmt::CREATE_DB: case Stmt::DROP_DB: needDb(s.type == Stmt::CREATE_DB ? "CREATE" : "DROP", s.table); break;
    case Stmt::USER:
      if (s.setName == "ALTER" && (s.table.empty() || s.table == tUser)) break;  // own password
      super();
      break;
    case Stmt::PIPELINE: case Stmt::BACKUP: case Stmt::SNAPSHOT: case Stmt::DROP_PLANCACHE: super(); break;
    default: break;
  }
}

ResultSet Engine::doUser(Stmt& s) {
  std::lock_guard<std::mutex> g(usersMu_);
  if (s.setName.rfind("GRANT", 0) == 0 || s.setName.rfind("REVOKE", 0) == 0) {
    const bool grant = s.setName[0] == 'G', role = s.setName.find("_ROLE") != std::string::npos;
    for (auto& who : s.grantees) {
      if (!users_.count(who)) throw SqlError("Can't find any matching row in the user table for '" + who + "'", 1133);
      for (auto& pv : s.names) {
        std::string e = role ? "@" + pv : pv + "|" + s.path;
        if (role) {
          std::string r = pv;
          if (!r.empty() && (r[0] == '\'' || r[0] == '"')) r = r.substr(1, r.size() - 2);
          if (!users_.count(lower(r)) && !users_.count(r)) throw SqlError("Unknown role '" + r + "'", 3523);
          e = "@" + (users_.count(r) ? r : lower(r));
        }
        if (grant) grants_[who].insert(e);
        else if (!role && pv == "ALL") {  // REVOKE ALL ON scope: every privilege on that scope
          for (auto it = grants_[who].begin(); it != grants_[who].end();)
            it = it->size() > s.path.size() && it->substr(it->size() - s.path.size() - 1) == "|" + s.path ? grants_[who].erase(it) : std::next(it);
        } else grants_[who].erase(e);
      }
    }
    return okResult(0);
  }
  const std::string u = s.table.empty() ? tUser : s.table;
  // Stored hash: SHA1(SHA1(pw)) (mysql_native_password) followed by SHA256(SHA256(pw)) (caching_sha2_password).
  const std::string hash = s.path == "\x01" || s.path.empty() ? "" : s.path == "\x03role" ? "!role" : sha1(sha1(s.path)) + sha256(sha256(s.path));
  if (s.setName == "CREATE") {
    if (users_.count(u)) {
      if (s.ifNotExists) return okResult(0);
      throw SqlError("Operation CREATE USER failed for '" + u + "'@'%'", 1396);
    }
    users_[u] = hash;
  } else if (s.setName == "DROP") {
    if (!users_.erase(u) && !s.ifExists) throw SqlError("Operation DROP USER failed for '" + u + "'@'%'", 1396);
    grants_.erase(u);
    for (auto& [k, v] : grants_) v.erase("@" + u);
  } else {
    if (!users_.count(u)) throw SqlError("Operation ALTER USER failed for '" + u + "'@'%'", 1396);
    if (s.path != "\x01") users_[u] = hash;
  }
  return okResult(0);
}

// BACKUP DATABASE db TO {'dir' | S3 'bucket/prefix' ..}: a consistent snapshot of the database
// (<db>.backup) plus a manifest with the WAL position it corresponds to. RESTORE DATABASE db [AS new]
// FROM .. [UNTIL TIMESTAMP 'ts'] loads it and, with UNTIL, replays the archived WAL (--wal-archive)
// and the live WAL for that database up to the time given (point-in-time restore).
static std::string jsonEsc(const std::string& x) {
  std::string o;
  for (char ch : x) {
    if (ch == '"' || ch == '\\') o += '\\';
    o += ch;
  }
  return o;
}

ResultSet Engine::doBackup(Stmt& s) {
  const std::string db = lower(s.table);
  const bool s3 = s.setValue == "S3";
  const std::string stage = (dir_.empty() ? std::string("/tmp/warpstore") : dir_) + "/backup-stage-" + std::to_string(nowUs());
  struct Cleanup {
    std::vector<std::string> files;
    std::string dir;
    ~Cleanup() {
      for (auto& f : files) unlink(f.c_str());
      if (!dir.empty()) rmdir(dir.c_str());
    }
  } cleanup;
  std::string prefix = s.path;  // S3: bucket/prefix
  while (!prefix.empty() && prefix.back() == '/') prefix.pop_back();
  const std::string bucket = s3 ? s3Bucket(prefix) : "";
  const std::string keyBase = s3 ? (prefix.size() > bucket.size() ? prefix.substr(bucket.size() + 1) + "/" : "") + db + ".backup" : "";
  auto localFile = [&] {
    if (s3) {
      mkdirs(stage);
      cleanup.dir = stage;
      return stage + "/" + db + ".backup";
    }
    if (s.path.size() > 7 && s.path.compare(s.path.size() - 7, 7, ".backup") == 0) return s.path;
    struct stat st;
    if (s.setName == "BACKUP" && stat(s.path.c_str(), &st) != 0) mkdirs(s.path);
    return s.path + "/" + db + ".backup";
  };
  if (s.setName == "BACKUP") {
    {
      std::lock_guard<std::mutex> g(cat_.mu);
      if (!cat_.dbs.count(db)) throw SqlError("Unknown database '" + s.table + "'", 1049);
    }
    if (tTxn.active) commitTxn();
    const std::string file = localFile();
    std::pair<uint64_t, int64_t> pos{0, 0};
    auto t0 = Clock::now();
    writeSnapshot(file, db, nullptr, &pos);
    const int64_t now = nowUs();
    const std::string man = "{\"format\":\"warpstore-backup-1\",\"database\":\"" + jsonEsc(db) + "\",\"created_us\":" + std::to_string(now) +
                            ",\"created\":\"" + Value::Ts(now).toString() + "\",\"wal_gen\":" + std::to_string(pos.first) +
                            ",\"wal_offset\":" + std::to_string(pos.second) + ",\"data_dir\":\"" + jsonEsc(dir_) + "\"}\n";
    {
      FILE* f = fopen((file + ".manifest").c_str(), "wb");
      if (!f) throw SqlError("cannot write " + file + ".manifest", 1105);
      fwrite(man.data(), 1, man.size(), f);
      fflush(f);
      fdatasync(fileno(f));
      fclose(f);
    }
    if (s3) {
      cleanup.files = {file, file + ".manifest", file + ".users"};
      s3PutFile(bucket, keyBase, file, s.config, s.credentials);
      s3PutFile(bucket, keyBase + ".manifest", file + ".manifest", s.config, s.credentials);
    } else {
      unlink((file + ".users").c_str());  // users / grants are server-wide: not part of a database backup
    }
    struct stat st;
    stat(s3 ? (stage + "/" + db + ".backup").c_str() : file.c_str(), &st);
    return okResult(0, "backup of " + db + " written to " + (s3 ? "s3://" + bucket + "/" + keyBase : file) + " (" +
                           std::to_string(st.st_size) + " bytes, " + fmtMs(msSince(t0)) + " ms, WAL position " + std::to_string(pos.first) + ":" +
                           std::to_string(pos.second) + ")");
  }
  // RESTORE
  const std::string target = s.asName.empty() ? db : lower(s.asName);
  {
    std::lock_guard<std::mutex> g(cat_.mu);
    for (auto& [k, t] : cat_.tables)
      if (lower(t->db) == target) throw SqlError("Can't restore: database '" + target + "' has tables", 1007);
  }
  const std::string file = localFile();
  if (s3) {
    cleanup.files = {file, file + ".manifest"};
    s3GetFile(bucket, keyBase, file, s.config, s.credentials);
    try {
      s3GetFile(bucket, keyBase + ".manifest", file + ".manifest", s.config, s.credentials);
    } catch (std::exception&) {
    }
  }
  if (access(file.c_str(), R_OK) != 0) throw SqlError("backup file " + file + " not found", 1105);
  std::string man;
  if (FILE* f = fopen((file + ".manifest").c_str(), "rb")) {
    char buf[4096];
    man.assign(buf, fread(buf, 1, sizeof buf, f));
    fclose(f);
  }
  int64_t untilUs = 0;
  if (!s.until.empty()) {
    if (!parseDateTime(s.until, untilUs)) throw SqlError("bad UNTIL TIMESTAMP '" + s.until + "'", 1525);
    if (man.empty()) throw SqlError("UNTIL needs the backup's manifest (" + file + ".manifest)", 1105);
    if (std::atoll(jsonField(man, "created_us").c_str()) > untilUs)
      throw SqlError("UNTIL TIMESTAMP is before the backup was taken (" + jsonField(man, "created") + ")", 1105);
  }
  auto t0 = Clock::now();
  const bool saved = replaying_;
  replaying_ = true;  // the restore itself is not logged statement by statement ...
  int64_t replayed = 0;
  try {
    loadSnapshot(file, db, target, false);
    if (untilUs)
      replayed = replayFrom(db, target, std::strtoull(jsonField(man, "wal_gen").c_str(), nullptr, 10), std::atoll(jsonField(man, "wal_offset").c_str()), untilUs);
  } catch (...) {
    replaying_ = saved;
    throw;
  }
  replaying_ = saved;
  if (!dir_.empty()) doSnapshot();  // ... so make it durable with a checkpoint
  return okResult(0, "restored " + target + " from " + (s3 ? "s3://" + bucket + "/" + keyBase : file) +
                         (untilUs ? " + " + std::to_string(replayed) + " WAL records up to " + s.until : "") + " in " + fmtMs(msSince(t0)) + " ms");
}

// Replays the WAL from {gen, off} (archive files, then the live WAL) for database db (as asDb) up to
// time untilUs. Returns the number of statements applied.
int64_t Engine::replayFrom(const std::string& db, const std::string& asDb, uint64_t gen, int64_t off, int64_t untilUs) {
  std::map<uint64_t, std::string> files;
  if (!gWalArchive.empty())
    if (DIR* dp = opendir(gWalArchive.c_str())) {
      while (dirent* e = readdir(dp)) {
        unsigned long long g;
        if (sscanf(e->d_name, "wal-%llu.log", &g) == 1 && g >= gen) files[g] = gWalArchive + "/" + e->d_name;
      }
      closedir(dp);
    }
  uint64_t liveGen;
  {
    std::lock_guard<std::mutex> g(walMu_);
    liveGen = walGen_;
  }
  if (!dir_.empty() && liveGen >= gen && !files.count(liveGen)) files[liveGen] = dir_ + "/wal.log";
  if (files.empty() || files.begin()->first != gen)
    throw SqlError("point-in-time restore needs WAL generation " + std::to_string(gen) + " (is --wal-archive set to the directory with the archived WALs?)", 1105);
  for (auto it = files.begin(); std::next(it) != files.end(); ++it)
    if (std::next(it)->first != it->first + 1)
      throw SqlError("WAL generation " + std::to_string(it->first + 1) + " is missing from the archive", 1105);
  int64_t applied = 0;
  const std::string savedDb = currentDb();
  auto apply = [&](const std::string& rec) {
    size_t sep = rec.find('\x1f');
    const std::string recDb = sep == std::string::npos ? "" : rec.substr(0, sep);
    std::string sql = sep == std::string::npos ? rec : rec.substr(sep + 1);
    Stmt st;
    try {
      st = parseSql(sql);
    } catch (std::exception&) {
      return;
    }
    const size_t dot = st.table.find('.');
    const std::string tdb = st.type == Stmt::CREATE_DB || st.type == Stmt::DROP_DB ? lower(st.table)
                            : dot != std::string::npos                             ? lower(st.table.substr(0, dot))
                                                                                   : lower(recDb);
    if (tdb != db || st.type == Stmt::CREATE_DB) return;
    if (asDb != db && dot != std::string::npos) {  // db.table in the statement: point it at the new name
      const size_t at = lower(sql).find(db + ".");
      if (at != std::string::npos) sql.replace(at, db.size(), asDb);
    }
    if (st.type == Stmt::DROP_DB) sql = "DROP DATABASE " + asDb;
    currentDb() = lower(recDb) == db ? asDb : recDb;
    try {
      execute(sql);
      ++applied;
    } catch (std::exception& e) {
      fprintf(stderr, "[warpstore] point-in-time restore: %s: %s\n", e.what(), sql.substr(0, 200).c_str());
    }
  };
  bool done = false;
  for (auto& [g, path] : files) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw SqlError("cannot read " + path, 1105);
    char hdr[16];
    if (fread(hdr, 1, 16, f) != 16 || memcmp(hdr, "GDBWAL3\n", 8) != 0) {
      fclose(f);
      throw SqlError(path + " is not a format-3 WAL", 1105);
    }
    fseeko(f, g == gen ? std::max<int64_t>(off, 16) : 16, SEEK_SET);
    for (;;) {
      uint32_t len, crc;
      if (fread(&len, 4, 1, f) != 1 || fread(&crc, 4, 1, f) != 1 || len == 0) break;
      std::string rec(len, '\0');
      if (fread(rec.data(), 1, len, f) != len || (uint32_t)crc32(0, (const Bytef*)rec.data(), (uInt)len) != crc) break;
      if (rec.rfind("\x1eTS", 0) == 0 && rec.size() >= 11) {
        int64_t us;
        memcpy(&us, rec.data() + 3, 8);
        if (us > untilUs) {
          done = true;
          break;
        }
      } else if (rec.rfind("\x1eTXN", 0) == 0) {
        for (size_t i = 4; i + 4 <= rec.size();) {
          uint32_t l;
          memcpy(&l, rec.data() + i, 4);
          i += 4;
          if (i + l > rec.size()) break;
          apply(rec.substr(i, l));
          i += l;
        }
      } else {
        apply(rec);
      }
    }
    fclose(f);
    if (done) break;
  }
  currentDb() = savedDb;
  return applied;
}


void Engine::connectionClosed() {
  if (tTxn.active) rollbackTxn();
  for (auto& n : tTempTables) {
    try {
      execute("DROP TABLE IF EXISTS " + n);
    } catch (...) {
    }
  }
  tTempTables.clear();
}

void Engine::loadSnapshot(const std::string& path, const std::string& fromDb, const std::string& toDb, bool withUsers) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return;
  auto mapDb = [&](const std::string& d) {
    if (fromDb.empty() && lower(d) == "gpudb") return std::string("warpstore");  // (data dirs from before the rename)
    return !fromDb.empty() && lower(d) == lower(fromDb) ? lower(toDb) : d;
  };
  auto t0 = Clock::now();
  const std::string savedDb = currentDb();
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "GDBSNAP", 7) != 0 || magic[7] < '1' || magic[7] > '5') {
    fclose(f);
    throw SqlError("bad snapshot file " + path, 1105);
  }
  std::string whole;  // the decompressed snapshot, when it was checked first (see below)
  if (magic[7] >= '2') {
    // Decompress it all first, so the zstd checksum is verified before anything is parsed (a
    // damaged file must not be half-loaded, or send the parser off on a garbage count). Very large
    // ones (backups with their data inline) are read as a stream; the checksum is then checked at
    // the end.
    const long at = ftell(f);
    ZSTD_DCtx* dz = ZSTD_createDCtx();
    std::vector<char> in(1 << 20);
    std::string out(1 << 20, '\0');
    size_t used = 0, rc = 1;
    bool ok = true, big = false;
    for (size_t got; ok && !big && (got = fread(in.data(), 1, in.size(), f)) > 0;) {
      ZSTD_inBuffer ib{in.data(), got, 0};
      while (ib.pos < ib.size) {
        if (used == out.size()) {
          if (out.size() >= ((size_t)1 << 30)) { big = true; break; }
          out.resize(out.size() * 2);
        }
        ZSTD_outBuffer ob{out.data() + used, out.size() - used, 0};
        rc = ZSTD_decompressStream(dz, &ob, &ib);
        if (ZSTD_isError(rc)) { ok = false; break; }
        used += ob.pos;
      }
    }
    ZSTD_freeDCtx(dz);
    if (!ok || (!big && rc != 0)) {
      fclose(f);
      throw SqlError("corrupt snapshot " + path + (ok ? " (truncated)" : " (checksum mismatch or damaged data)"), 1105);
    }
    if (big) {
      fseek(f, at, SEEK_SET);
      f = zstdReader(f);
    } else {
      fclose(f);
      out.resize(used);
      whole = std::move(out);
      f = fmemopen(whole.data(), std::max<size_t>(whole.size(), 1), "rb");
      if (!f) throw SqlError("cannot read snapshot " + path, 1105);
      if (whole.empty()) fgetc(f);  // (an empty stream: EOF at once)
    }
  }
  const bool columnar = magic[7] >= '3';
  uint64_t n;
  getU64(f, n);
  for (uint64_t i = 0; i < n; ++i) {
    std::string d;
    getStr(f, d);
    if (d.size() > 5 && d.compare(d.size() - 5, 5, "\x1fSYNC") == 0) {
      d.resize(d.size() - 5);
      cat_.syncDbs.insert(mapDb(d));
    }
    cat_.dbs.insert(mapDb(d));
  }
  getU64(f, n);
  for (uint64_t i = 0; i < n; ++i) {
    std::string k, v;
    getStr(f, k);
    getStr(f, v);
    const size_t dot = k.find('.');
    if (dot != std::string::npos) k = mapDb(k.substr(0, dot)) + k.substr(dot);
    cat_.views[k] = v;
  }
  getU64(f, n);
  const uint64_t ntables = n;
  int64_t total = 0;
  for (uint64_t i = 0; i < n; ++i) {
    std::string db, ddl;
    uint64_t next, rows;
    getStr(f, db);
    getStr(f, ddl);
    getU64(f, next);
    getU64(f, rows);
    db = mapDb(db);
    currentDb() = db;
    Stmt cs = parseSql(ddl);
    doCreate(cs);
    auto t = cat_.get(db + "." + cs.table);
    std::unique_lock<FairSharedMutex> lk(t->mu);
    std::vector<Value> row(t->cols.size());
    const std::string dictDir = dir_ + "/dicts";
    const auto tt0 = Clock::now();
    if (columnar) readColumnar(f, *t, magic[7] >= '4', magic[7] >= '5' ? &dictDir : nullptr), rows = 0;  // (rows: counted below from the table)
    if (envSetting("TRACE")) fprintf(stderr, "[trace] snapshot: table %s in %.0f ms\n", t->name.c_str(), msSince(tt0));
    for (uint64_t r = 0; r < rows; ++r) {
      for (auto& v : row) getVal(f, v);
      t->appendRow(row);
    }
    t->autoNext = (int64_t)next;
    t->flush(true);
    total += columnar ? t->liveRows() : (int64_t)rows;
  }
  if (getU64(f, n))
    for (uint64_t i = 0; i < n; ++i) {
      auto p = std::make_shared<Pipeline>();
      for (auto* x : {&p->name, &p->db, &p->table, &p->glob, &p->delim, &p->cols}) getStr(f, *x);
      p->db = mapDb(p->db);
      uint64_t skip, want, nd;
      getU64(f, skip);
      getU64(f, want);
      getU64(f, nd);
      p->skip = (int64_t)skip;
      p->wantRun = want != 0;
      for (uint64_t k = 0; k < nd; ++k) {
        std::string d;
        getStr(f, d);
        p->done.insert(d);
      }
      pipes_[lower(p->name)] = p;
    }
  if (magic[7] >= '2' && whole.empty()) {  // (streamed: the end of the zstd stream, where its checksum is verified)
    char extra;
    const size_t k = fread(&extra, 1, 1, f);
    if (k != 0 || ferror(f)) {
      fclose(f);
      throw SqlError("corrupt snapshot " + path + " (checksum mismatch or trailing data)", 1105);
    }
  }
  fclose(f);
  if (FILE* uf = withUsers ? fopen((path + ".users").c_str(), "rb") : nullptr) {
    uint64_t nu;
    if (getU64(uf, nu))
      for (uint64_t i = 0; i < nu; ++i) {
        std::string u, h;
        if (!getStr(uf, u) || !getStr(uf, h)) break;
        std::lock_guard<std::mutex> g(usersMu_);
        users_[u] = h;
      }
    if (getU64(uf, nu))
      for (uint64_t i = 0; i < nu; ++i) {
        std::string d;
        if (!getStr(uf, d)) break;
        try {
          ResultSet rr;
          routineStatement(d, rr);
        } catch (std::exception& ex) {
          fprintf(stderr, "[warpstore] routine restore failed: %s\n", ex.what());
        }
      }
    if (getU64(uf, nu))
      for (uint64_t i = 0; i < nu; ++i) {
        std::string u, e;
        uint64_t ne;
        if (!getStr(uf, u) || !getU64(uf, ne)) break;
        std::lock_guard<std::mutex> g(usersMu_);
        for (uint64_t j = 0; j < ne && getStr(uf, e); ++j) grants_[u].insert(e);
      }
    fclose(uf);
  }
  currentDb() = savedDb;
  fprintf(stderr, "[warpstore] loaded snapshot: %llu tables, %lld rows in %.0f ms\n", (unsigned long long)ntables, (long long)total, msSince(t0));
}


}  // namespace gdb

