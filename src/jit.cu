#include "jit.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include "common.h"
#include "gpu.h"
#include "jit_sources.h"  // generated at build time: devtypes.h and kernels.cuh as strings

namespace gdb {

namespace {

std::atomic<int> gDefaultMode{JIT_ON};
thread_local int tModeSet = -1;  // -1: session hasn't SET it, use the server default
std::mutex gMu;
std::map<std::string, std::shared_ptr<JitProgram>> gCache;  // key: kernel kind + generated evaluator source
std::vector<std::shared_ptr<JitProgram>> gById;
thread_local std::string tSql;
thread_local JitQueryInfo tInfo;
thread_local std::vector<JitProgram*> tUsed;
std::string gDiskDir;  // persistent plan cache directory ("" = disabled)

// ---------------------------------------------------------------- persistent plan cache
// Compiled machine code is stored on disk keyed by a hash of the full kernel source, the target
// architecture, the NVRTC version and the options, so a query shape compiles once ever, not once
// per server start. The file also holds the source, which is compared on load (no false hits), and a
// checksum of the machine code, so a corrupted file is recompiled instead of run.

uint64_t fnv64(const std::string& s, uint64_t h) {
  for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ull;
  return h;
}

std::string diskPath(const std::string& key) {
  char name[48];
  snprintf(name, sizeof name, "%016llx%016llx.plan", (unsigned long long)fnv64(key, 0xcbf29ce484222325ull),
           (unsigned long long)fnv64(key, 0x84222325cbf29ce4ull));
  return gDiskDir + "/" + name;
}

bool diskLoad(const std::string& path, const std::string& key, std::string& image) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::string magic(8, '\0');
  uint64_t kl = 0, il = 0;
  f.read(magic.data(), 8);
  f.read((char*)&kl, 8);
  if (!f || magic != "GDBPLAN2" || kl != key.size()) return false;
  std::string k(kl, '\0');
  f.read(k.data(), kl);
  f.read((char*)&il, 8);
  if (!f || k != key || il > (1ull << 30)) return false;
  image.assign(il, '\0');
  f.read(image.data(), il);
  uint64_t sum = 0;
  f.read((char*)&sum, 8);
  if (!f || sum != fnv64(image, 0xcbf29ce484222325ull)) {
    fprintf(stderr, "[gpu-db] plan cache file %s is corrupt; recompiling\n", path.c_str());
    return false;
  }
  return true;
}

void diskStore(const std::string& path, const std::string& key, const std::string& image) {
  const std::string tmp = path + ".tmp" + std::to_string((unsigned long long)std::hash<std::thread::id>{}(std::this_thread::get_id()));
  {
    std::ofstream f(tmp, std::ios::binary);
    const uint64_t kl = key.size(), il = image.size();
    f.write("GDBPLAN2", 8);
    f.write((const char*)&kl, 8);
    f.write(key.data(), kl);
    f.write((const char*)&il, 8);
    f.write(image.data(), il);
    const uint64_t sum = fnv64(image, 0xcbf29ce484222325ull);
    f.write((const char*)&sum, 8);
    if (!f) return;
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);  // atomic publish; concurrent writers just race to the same content
}

// ---------------------------------------------------------------- bytecode -> typed CUDA C++

struct Sym {
  std::string e;
  bool isInt;
};

std::string asD(const Sym& s) { return s.isInt ? "((double)" + s.e + ")" : s.e; }
std::string asI(const Sym& s) { return s.isInt ? s.e : "((long long)" + s.e + ")"; }

// Rebuilds the expression tree from the stack bytecode, typing every node: integer columns
// (INT/BIGINT/DATE/dictionary codes) and integer literals stay in 64-bit integer arithmetic, so a
// filter like `shares > 500` never touches FP64 (1/64 rate on consumer GPUs).
struct Translator {
  const ExprEnv& env;
  int ncols;
  std::set<int> cols, iconsts, dconsts;

  Sym run(Prog p) {
    std::vector<Sym> st;
    for (int pc = p.start; pc < p.start + p.len; ++pc) {
      const Instr in = env.code[pc];
      auto pop = [&]() {
        Sym s = st.back();
        st.pop_back();
        return s;
      };
      switch (in.op) {
        case OP_COL: {
          if (in.arg >= ncols) throw SqlError("codegen: column slot out of range", 1105);
          cols.insert(in.arg);
          const int t = env.cols[in.arg].t;
          const std::string n = std::to_string(in.arg);
          // Raw, or an out-of-line raw-or-compressed load when the table has compressed segments of this
          // column (inlined, every such load doubles the loop versions NVRTC generates).
          const std::string c = !env.cols[in.arg].enc ? (t == 2 ? "c" + n + "[r]" : "((long long)c" + n + "[r])")
                              : t == 2 ? "gdb::colDblNI(v" + n + ",r)" : "gdb::colIntNI(v" + n + ",r)";
          if (t == 2) st.push_back({c, false});
          else if (t == 3) st.push_back({"gdb::nullIfI32(" + c + ")", false});
          else if (t == 4) st.push_back({"gdb::nullIfI64(" + c + ")", false});
          else if (t == 6)
            st.push_back({"gdb::decOf(" + c + "," + std::to_string((long long)pow10i(env.cols[in.arg].scale)) + ".0)", false});
          else st.push_back({c, true});
          break;
        }
        case OP_COLRAW: {
          cols.insert(in.arg);
          const std::string n = std::to_string(in.arg);
          const std::string c = !env.cols[in.arg].enc ? "((long long)c" + n + "[r])" : "gdb::colIntNI(v" + n + ",r)";
          if (env.cols[in.arg].t == 6 && !(in.isInt)) st.push_back({"gdb::nullIfI64(" + c + ")", false});
          else st.push_back({c, true});
          break;
        }
        case OP_CONST:
          if (in.isInt) {
            iconsts.insert(in.arg);
            st.push_back({"ik" + std::to_string(in.arg), true});
          } else {
            dconsts.insert(in.arg);
            st.push_back({"k" + std::to_string(in.arg), false});
          }
          break;
        case OP_ADD: case OP_SUB: case OP_MUL: {
          Sym b = pop(), a = pop();
          const char* o = in.op == OP_ADD ? "+" : in.op == OP_SUB ? "-" : "*";
          if (a.isInt && b.isInt)  // checked: an overflow sets env.ovf (BIGINT value is out of range)
            st.push_back({std::string(in.op == OP_ADD ? "gdb::iaddc(" : in.op == OP_SUB ? "gdb::isubc(" : "gdb::imulc(") + a.e + "," + b.e + ",env.ovf)", true});
          else st.push_back({"(" + asD(a) + o + asD(b) + ")", false});
          break;
        }
        case OP_DIV: {
          Sym b = pop(), a = pop();
          st.push_back({"gdb::divz(" + asD(a) + "," + asD(b) + ")", false});
          break;
        }
        case OP_MOD: {
          Sym b = pop(), a = pop();
          st.push_back({"fmod(" + asD(a) + "," + asD(b) + ")", false});
          break;
        }
        case OP_NEG: {
          Sym a = pop();
          st.push_back({"(-" + a.e + ")", a.isInt});
          break;
        }
        case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE: {
          Sym b = pop(), a = pop();
          static const char* ops[] = {"==", "!=", "<", "<=", ">", ">="};
          const char* o = ops[in.op - OP_EQ];
          // Integer operands can't be NULL: plain comparison. Otherwise three-valued (NaN = NULL).
          if (a.isInt && b.isInt) st.push_back({"((long long)(" + a.e + o + b.e + "))", true});
          else st.push_back({"gdb::cmp3(" + std::to_string(in.op - OP_EQ) + "," + asD(a) + "," + asD(b) + ")", false});
          break;
        }
        case OP_AND: case OP_OR: {
          Sym b = pop(), a = pop();
          if (a.isInt && b.isInt)
            st.push_back({"((long long)((" + a.e + "!=0)" + (in.op == OP_AND ? "&&" : "||") + "(" + b.e + "!=0)))", true});
          else st.push_back({std::string(in.op == OP_AND ? "gdb::and3(" : "gdb::or3(") + asD(a) + "," + asD(b) + ")", false});
          break;
        }
        case OP_NOT: {
          Sym a = pop();
          if (a.isInt) st.push_back({"((long long)(" + a.e + "==0))", true});
          else st.push_back({"gdb::not3(" + a.e + ")", false});
          break;
        }
        case OP_INSET: {
          Sym a = pop();
          const std::string s = std::to_string(in.arg);
          if (a.isInt) st.push_back({"gdb::gdbInset(env.sets[" + s + "],env.setSize[" + s + "]," + a.e + ")", true});
          else st.push_back({"gdb::insetD(env.sets[" + s + "],env.setSize[" + s + "]," + a.e + ")", false});
          break;
        }
        case OP_SELECT: {
          Sym e = pop(), t = pop(), c = pop();
          const std::string cond = c.isInt ? "(" + c.e + "!=0)" : "gdb::truthy(" + c.e + ")";
          if (t.isInt && e.isInt) st.push_back({"(" + cond + "?" + t.e + ":" + e.e + ")", true});
          else st.push_back({"(" + cond + "?" + asD(t) + ":" + asD(e) + ")", false});
          break;
        }
        case OP_YEAR: {
          Sym a = pop();
          if (a.isInt) st.push_back({"gdb::gdbYear(" + a.e + ")", true});
          else st.push_back({"gdb::gdbCall1(gdb::FN_DT_YEAR,gdb::gdbCall1(gdb::FN_DATE_TO_DT," + a.e + "))", false});
          break;
        }
        case OP_ISNULL: {
          Sym a = pop();
          st.push_back({a.isInt ? "0ll" : "((long long)gdb::isNullD(" + a.e + "))", true});  // once: two copies may round differently (FMA)
          break;
        }
        case OP_COALESCE: {
          Sym b = pop(), a = pop();
          if (a.isInt) st.push_back(a);
          else st.push_back({"gdb::coalesce2(" + a.e + "," + asD(b) + ")", false});
          break;
        }
        case OP_NULLIF: {
          Sym b = pop(), a = pop();
          st.push_back({"gdb::nullif2(" + asD(a) + "," + asD(b) + ")", false});
          break;
        }
        case OP_CALL1: {
          Sym a = pop();
          st.push_back({"gdb::gdbCall1(" + std::to_string(in.arg) + "," + asD(a) + ")", false});
          break;
        }
        case OP_CALL2: {
          Sym b = pop(), a = pop();
          st.push_back({"gdb::gdbCall2(" + std::to_string(in.arg) + "," + asD(a) + "," + asD(b) + ")", false});
          break;
        }
        case OP_VDOT: case OP_VEUC: {
          const int slot = in.arg & 255, m = in.arg >> 8;
          cols.insert(slot);
          const std::string ms = std::to_string(m);
          st.push_back({std::string(in.op == OP_VDOT ? "gdb::vdotD(" : "gdb::veucD(") + "c" + std::to_string(slot) +
                            ",r,env.maps[" + ms + "],env.mapSize[" + ms + "])", false});
          break;
        }
        case OP_INSORTED: {
          Sym a = pop();
          const std::string s = std::to_string(in.arg);
          st.push_back({"gdb::insortedD(env.maps[" + s + "],env.mapSize[" + s + "]," + asD(a) + ")", false});
          break;
        }
        case OP_DICTMAP: {
          Sym a = pop();
          const std::string s = std::to_string(in.arg);
          st.push_back({"gdb::dictmapD(env.maps[" + s + "],env.mapSize[" + s + "]," + asD(a) + ")", false});
          break;
        }
        default: throw SqlError("codegen: unknown opcode", 1105);
      }
    }
    if (st.size() != 1) throw SqlError("codegen: malformed expression", 1105);
    return st[0];
  }
};

const char* ctypeOf(int t) {
  return t == 0 || t == 3 ? "const int*" : t == 1 || t == 4 || t == 6 ? "const long long*" : t == 5 ? "const float*" : "const double*";
}

std::string genEvaluator(const ExprEnv& env, int ncols, const AggPlan& plan) {
  Translator tr{env, ncols, {}, {}, {}};
  Sym where = plan.hasWhere ? tr.run(plan.where) : Sym{"1", true};
  std::vector<Sym> keys, aggs;
  for (int k = 0; k < plan.nkeys; ++k) keys.push_back(tr.run(plan.keys[k]));
  for (int a = 0; a < plan.naggs; ++a) aggs.push_back(plan.ops[a] == AGG_COUNT ? Sym{"1ll", true} : tr.run(plan.aggs[a]));
  std::ostringstream o;
  o << "struct JitEv {\n  static constexpr int kDensePriv = " << (int)plan.densePriv << ";\n";
  o << "  __device__ uint8_t dkind(int a, const gdb::DenseSpec&) const { switch (a) {";
  for (int a = 0; a < plan.naggs; ++a) {
    const int op = plan.ops[a];
    const bool in = plan.aggInt[a];
    const int k = op == AGG_COUNT ? DK_COUNT : op == AGG_SUM ? (in ? DK_SUM_I : DK_SUM_D) : op == AGG_MIN ? (in ? DK_MIN_I : DK_MIN_D)
                                                                                                         : (in ? DK_MAX_I : DK_MAX_D);
    o << " case " << a << ": return " << k << ";";
  }
  o << " } return 0; }\n";
  o << "  static constexpr int kKeys = " << std::max(plan.nkeys, 1) << ", kAggs = " << std::max(plan.naggs, 1)
    << ";\n  const gdb::ExprEnv& env;\n";
  for (int c : tr.cols)
    o << "  " << ctypeOf(env.cols[c].t) << " c" << c << ";\n  gdb::ColView v" << c << ";\n  int e" << c << ";\n  long long b" << c << ";\n";
  for (int k : tr.iconsts) o << "  long long ik" << k << ";\n";
  for (int k : tr.dconsts) o << "  double k" << k << ";\n";
  o << "  __device__ JitEv(const gdb::ExprEnv& e, const gdb::AggPlan&) : env(e) {\n";
  for (int k : tr.iconsts) o << "    ik" << k << " = gdb::bitsOf(e.consts[" << k << "]);\n";
  for (int k : tr.dconsts) o << "    k" << k << " = e.consts[" << k << "];\n";
  o << "  }\n  __device__ void bind(const gdb::ColView* c) {\n";
  for (int c : tr.cols)
    o << "    c" << c << " = (" << ctypeOf(env.cols[c].t) << ")c[" << c << "].p; v" << c << " = c[" << c << "]; e" << c << " = c[" << c
      << "].enc; b" << c << " = e" << c << " && c[" << c << "].t != 2 ? *(const long long*)c[" << c << "].p : 0;\n";
  o << "  }\n";
  o << "  __device__ const gdb::ColView* envCols(const gdb::ExprEnv& e) const { return e.cols; }\n";
  o << "  __device__ bool keyIsInt(int k) const { switch (k) {";
  for (int k = 0; k < plan.nkeys; ++k) o << " case " << k << ": return " << (plan.keyInt[k] ? "true" : "false") << ";";
  o << " } return false; }\n";
  o << "  __device__ int nkeys() const { return " << plan.nkeys << "; }\n";
  o << "  __device__ int naggs() const { return " << plan.naggs << "; }\n";
  o << "  __device__ bool hasWhere() const { return " << (plan.hasWhere ? "true" : "false") << "; }\n";
  o << "  __device__ int op(int a) const { switch (a) {";
  for (int a = 0; a < plan.naggs; ++a) o << " case " << a << ": return " << (int)plan.ops[a] << ";";
  o << " } return 0; }\n";
  o << "  __device__ bool aggIsInt(int a) const { switch (a) {";
  for (int a = 0; a < plan.naggs; ++a) o << " case " << a << ": return " << (plan.aggInt[a] ? "true" : "false") << ";";
  o << " } return false; }\n";
  o << "  __device__ bool where(long long r) const { return " << (where.isInt ? where.e + " != 0" : "gdb::truthy(" + where.e + ")") << "; }\n";
  o << "  __device__ double key(int k, long long r) const { switch (k) {";
  for (size_t k = 0; k < keys.size(); ++k) o << "\n    case " << k << ": return " << asD(keys[k]) << ";";
  o << " } return 0; }\n";
  o << "  __device__ long long ikey(int k, long long r) const { switch (k) {";
  for (size_t k = 0; k < keys.size(); ++k) o << "\n    case " << k << ": return " << asI(keys[k]) << ";";
  o << " } return 0; }\n";
  o << "  __device__ double agg(int a, long long r) const { switch (a) {";
  for (size_t a = 0; a < aggs.size(); ++a) o << "\n    case " << a << ": return " << asD(aggs[a]) << ";";
  o << " } return 0; }\n";
  o << "  __device__ long long iagg(int a, long long r) const { switch (a) {";
  for (size_t a = 0; a < aggs.size(); ++a)  // (a nullable int column: its raw value, the kernel skips the NULL sentinel)
    o << "\n    case " << a << ": return "
      << (plan.aggNull[a] ? "gdb::colInt(v" + std::to_string(env.code[plan.aggs[a].start].arg) + ", r)" : asI(aggs[a])) << ";";
  o << " } return 0; }\n};\n";
  return o.str();
}

std::string stripPragmaOnce(std::string s) {
  for (size_t p; (p = s.find("#pragma once")) != std::string::npos;) s.erase(p, 12);
  return s;
}

void ckDrv(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    throw SqlError(std::string("CUDA driver error in ") + what + ": " + (msg ? msg : "?"), 1105);
  }
}

const char* kKernelMacro[] = {"GDB_KERNEL_NOKEYS", "GDB_KERNEL_GROUPED", "GDB_KERNEL_DENSE", "GDB_KERNEL_FLAG", "GDB_KERNEL_FLAGM"};
const char* kKernelName[] = {"gdb_aggNoKeys", "gdb_aggGrouped", "gdb_aggDense", "gdb_flag", "gdb_flagMulti"};

// Compiles the one kernel the plan needs; returns false (and logs) on failure.
// Full kernel source, options and persistent-cache key for a plan.
struct PlanBuild {
  std::string src, key, path;
  std::vector<std::string> opts;
  bool ptx = false;
};

PlanBuild planBuild(const JitProgram& p) {
  static const std::string prelude = stripPragmaOnce(kJitDevtypesSrc) + "\n" + stripPragmaOnce(kJitKernelsSrc) + "\n";
  PlanBuild b;
  b.src = prelude + p.source + kKernelMacro[p.fn_kind] + "(JitEv, extern \"C\")\n";
  // Experiment hooks: GPUDB_NVRTC_PTX=1 emits PTX for the driver to assemble; GPUDB_NVRTC_OPTS adds flags.
  b.ptx = getenv("GPUDB_NVRTC_PTX") != nullptr;
  b.opts = {std::string("--gpu-architecture=") + (b.ptx ? "compute_" : "sm_") + std::to_string(gpuArch()), "--std=c++17", "--device-int128",
            "-DNDEBUG"};
  if (const char* extra = getenv("GPUDB_NVRTC_OPTS")) {
    std::istringstream is(extra);
    for (std::string o; is >> o;) b.opts.push_back(o);
  }
  if (!gDiskDir.empty()) {
    int maj = 0, min = 0;
    nvrtcVersion(&maj, &min);
    b.key = "nvrtc " + std::to_string(maj) + "." + std::to_string(min) + "\n";
    for (auto& o : b.opts) b.key += o + "\n";
    b.key += b.src;
    b.path = diskPath(b.key);
  }
  return b;
}

void loadImage(JitProgram& p, const std::string& image) {
  cudaFree(nullptr);  // make sure this thread has the primary context current
  CUmodule mod;
  ckDrv(cuModuleLoadData(&mod, image.data()), "cuModuleLoadData");
  p.module = mod;
  CUfunction f;
  ckDrv(cuModuleGetFunction(&f, mod, kKernelName[p.fn_kind]), "cuModuleGetFunction");
  p.fn[p.fn_kind] = f;
}

// Loads the plan from the persistent cache (about 0.3 ms); false if it isn't there.
bool tryDisk(JitProgram& p) {
  if (gDiskDir.empty()) return false;
  auto t0 = std::chrono::steady_clock::now();
  PlanBuild b = planBuild(p);
  std::string image;
  if (!diskLoad(b.path, b.key, image)) return false;
  try {
    loadImage(p, image);
  } catch (const std::exception&) {
    return false;  // e.g. stale file for another driver: recompile
  }
  p.fromDisk = true;
  p.compileMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (getenv("GPUDB_TRACE"))
    fprintf(stderr, "[codegen] plan #%d %s: loaded from disk cache in %.1f ms\n", p.id, kKernelName[p.fn_kind], p.compileMs);
  return true;
}

// Compiles the one kernel the plan needs with NVRTC and stores it in the persistent cache;
// returns false (and logs) on failure.
bool compile(JitProgram& p) {
  using C = std::chrono::steady_clock;
  auto ms = [](C::time_point a) { return std::chrono::duration<double, std::milli>(C::now() - a).count(); };
  PlanBuild b = planBuild(p);
  auto t0 = C::now();
  if (const char* d = getenv("GPUDB_DUMP")) {
    FILE* f = fopen((std::string(d) + "/plan" + std::to_string(p.id) + ".cu").c_str(), "w");
    if (f) fwrite(b.src.data(), 1, b.src.size(), f), fclose(f);
  }
  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, b.src.c_str(), "gpudb_query.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) return false;
  std::vector<const char*> opts;
  for (auto& o : b.opts) opts.push_back(o.c_str());
  nvrtcResult rc = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
  const double tCompile = ms(t0);
  if (rc != NVRTC_SUCCESS) {
    size_t n = 0;
    nvrtcGetProgramLogSize(prog, &n);
    std::string log(n, '\0');
    nvrtcGetProgramLog(prog, log.data());
    fprintf(stderr, "[gpu-db] codegen failed, using the interpreter:\n%s\n", log.c_str());
    nvrtcDestroyProgram(&prog);
    return false;
  }
  size_t n = 0;
  std::string image;
  if (b.ptx) {
    nvrtcGetPTXSize(prog, &n);
    image.assign(n, '\0');
    nvrtcGetPTX(prog, image.data());
  } else {
    nvrtcGetCUBINSize(prog, &n);
    image.assign(n, '\0');
    nvrtcGetCUBIN(prog, image.data());
  }
  nvrtcDestroyProgram(&prog);
  if (!b.path.empty()) diskStore(b.path, b.key, image);
  auto t1 = C::now();
  loadImage(p, image);
  if (getenv("GPUDB_TRACE"))
    fprintf(stderr, "[codegen] plan #%d %s: nvrtc %.1f ms, load %.1f ms, %zu source bytes\n", p.id, kKernelName[p.fn_kind],
            tCompile, ms(t1), b.src.size());
  return true;
}

void compileAndPublish(const std::shared_ptr<JitProgram>& p) {
  auto t0 = std::chrono::steady_clock::now();
  bool ok = false;
  try {
    ok = compile(*p);
  } catch (const std::exception& e) {
    fprintf(stderr, "[gpu-db] codegen failed: %s\n", e.what());
  }
  p->compileMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  p->state.store(ok ? 1 : 2, std::memory_order_release);
}

}  // namespace

JitMode jitMode() { return (JitMode)(tModeSet >= 0 ? tModeSet : gDefaultMode.load()); }
bool jitEnabled() { return jitMode() != JIT_OFF; }
void jitSetMode(JitMode m) { tModeSet = m; }
void jitSetDefaultMode(JitMode m) { gDefaultMode = m; }

void jitSetCacheDir(const std::string& dir) {
  if (dir.empty()) {
    gDiskDir.clear();
    return;
  }
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  gDiskDir = ec ? "" : dir;
  if (ec) fprintf(stderr, "[gpu-db] plan cache directory %s unusable (%s); compiled plans stay in memory\n", dir.c_str(), ec.message().c_str());
}

bool progIsInt(const ExprEnv& env, Prog p) {
  Translator tr{env, kMaxCols, {}, {}, {}};
  return tr.run(p).isInt;
}

const JitProgram* jitGet(const ExprEnv& env, int ncols, const AggPlan& plan, JitFn fn, const std::string& kind) {
  const JitMode tMode = jitMode();
  if (tMode == JIT_OFF) return nullptr;
  std::string ev = genEvaluator(env, ncols, plan);
  const std::string key = std::string(kKernelMacro[fn]) + "\n" + ev;
  std::shared_ptr<JitProgram> p;
  bool fresh = false;
  {
    std::lock_guard<std::mutex> g(gMu);
    auto it = gCache.find(key);
    if (it != gCache.end()) {
      p = it->second;
    } else {
      p = std::make_shared<JitProgram>();
      p->id = (int)gById.size() + 1;
      p->fn_kind = fn;
      p->kind = kind;
      p->sql = tSql;
      p->source = ev;
      gById.push_back(p);
      gCache.emplace(key, p);
      fresh = true;
    }
  }
  if (fresh && tryDisk(*p)) {
    p->state.store(1, std::memory_order_release);  // compiled in an earlier run: ready right away
    ++tInfo.fromDisk;
  } else if (fresh) {
    if (tMode == JIT_SYNC) {
      compileAndPublish(p);
      if (p->state.load() == 1) {
        ++tInfo.compiled;
        tInfo.compileMs += p->compileMs;
      }
    } else {
      std::thread([p] { compileAndPublish(p); }).detach();  // interpret first, compile in background
    }
  } else if (tMode == JIT_SYNC && p->state.load(std::memory_order_acquire) == 0) {
    while (p->state.load(std::memory_order_acquire) == 0) std::this_thread::yield();  // compiled by another session
  }
  const int st = p->state.load(std::memory_order_acquire);
  if (st != 1) {
    if (st == 0) ++tInfo.background;
    return nullptr;
  }
  if (!fresh) ++tInfo.hits;
  std::lock_guard<std::mutex> g(gMu);
  if (std::find(tUsed.begin(), tUsed.end(), p.get()) == tUsed.end()) {
    tUsed.push_back(p.get());
    tInfo.ids.push_back(p->id);
  }
  return p.get();
}

void jitLaunch(const JitProgram* p, JitFn f, unsigned grid, unsigned block, size_t shmem, void** args) {
  if (shmem > 32 * 1024)  // dynamic + static > 48 KB needs the opt-in (up to 99 KB on sm_86)
    cuFuncSetAttribute((CUfunction)p->fn[f], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)shmem);
  ckDrv(cuLaunchKernel((CUfunction)p->fn[f], grid, 1, 1, block, 1, 1, (unsigned)shmem, CU_STREAM_PER_THREAD, args, nullptr),
        "cuLaunchKernel");
}

void jitBeginQuery(const std::string& sql) {
  tSql.clear();  // one line, single spaces (for SHOW PLANCACHE)
  for (char c : sql) {
    const bool sp = c == ' ' || c == '\n' || c == '\t' || c == '\r';
    if (sp && (tSql.empty() || tSql.back() == ' ')) continue;
    tSql += sp ? ' ' : c;
  }
  while (!tSql.empty() && tSql.back() == ' ') tSql.pop_back();
  tUsed.clear();
  tInfo = JitQueryInfo();
}

JitQueryInfo jitEndQuery(double queryMs) {
  std::lock_guard<std::mutex> g(gMu);
  for (JitProgram* p : tUsed) {
    ++p->execs;
    p->totalMs += queryMs;
    p->lastMs = queryMs;
  }
  tUsed.clear();
  return tInfo;
}

std::vector<PlanCacheRow> jitPlanCache() {
  std::lock_guard<std::mutex> g(gMu);
  std::vector<PlanCacheRow> r;
  for (auto& p : gById) {
    const int st = p->state.load();
    r.push_back({p->id, st == 1 ? p->kind : p->kind + (st == 0 ? " (compiling)" : " (compile failed)"),
                 st != 1 ? "" : p->fromDisk ? "disk" : "compiled", p->compileMs, p->execs,
                 p->execs ? p->totalMs / p->execs : 0, p->lastMs, p->source.size(), p->sql});
  }
  return r;
}

std::string jitSourceOf(int id) {
  std::lock_guard<std::mutex> g(gMu);
  for (auto& p : gById)
    if (p->id == id) return p->source;
  return "";
}

void jitClearCache() {
  // Modules are not unloaded: a query on another connection may still be running one.
  std::lock_guard<std::mutex> g(gMu);
  gById.clear();
  gCache.clear();
}

}  // namespace gdb
