// Query compilation: each query's expressions are translated from bytecode into
// typed CUDA C++, compiled at runtime with NVRTC and cached in a plan cache keyed by the generated
// source. Literal values are not part of the source (they are read from ExprEnv::consts), so the
// same query shape with different constants reuses the compiled plan.
#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include "devtypes.h"

namespace gdb {

enum JitFn { JF_NOKEYS = 0, JF_GROUPED = 1, JF_DENSE = 2, JF_FLAG = 3, JF_FLAGM = 4 };

struct JitProgram {
  int id = 0;
  JitFn fn_kind = JF_FLAG;
  std::string kind;  // "aggregate (dense)", "filter", ...
  std::string sql;   // the query that first compiled it
  std::string source;
  double compileMs = 0;
  void* module = nullptr;  // CUmodule
  void* fn[5] = {};        // CUfunction per JitFn (only fn[fn_kind] is compiled)
  std::atomic<int> state{0};  // 0 = compiling, 1 = ready, 2 = failed
  bool fromDisk = false;      // loaded from the persistent plan cache instead of compiled
  // Statistics (guarded by the cache mutex).
  int64_t execs = 0;
  double totalMs = 0, lastMs = 0;
};

// Per session (connection thread): SET gpudb_codegen = ON | SYNC | OFF.
//   ON   (default) interpret first: a new query shape runs interpreted while its kernel compiles in
//        the background; later executions use the compiled plan (SingleStore's interpret_first).
//   SYNC compile before the first execution (benchmarks).   OFF  always interpret.
enum JitMode { JIT_OFF = 0, JIT_ON = 1, JIT_SYNC = 2 };
bool jitEnabled();
JitMode jitMode();
void jitSetMode(JitMode m);         // this session
void jitSetDefaultMode(JitMode m);  // server default (--codegen)
// Persistent plan cache directory (--plancache-dir); "" keeps compiled plans in memory only.
void jitSetCacheDir(const std::string& dir);

// Compiled kernels for the expressions of `plan` over the first `ncols` columns of env (their
// physical types are read from env.cols[i].t). From the plan cache, or compiled now. Returns
// nullptr when codegen is off or compilation failed (callers fall back to the interpreter).
const JitProgram* jitGet(const ExprEnv& env, int ncols, const AggPlan& plan, JitFn fn, const std::string& kind);
void jitLaunch(const JitProgram* p, JitFn f, unsigned grid, unsigned block, size_t shmem, void** args);

// Static type of an expression: true when it is integer-valued (columns typed via env.cols[i].t).
bool progIsInt(const ExprEnv& env, Prog p);

// Per-query bookkeeping, for SHOW PLANCACHE and PROFILE.
struct JitQueryInfo {
  int compiled = 0, hits = 0, fromDisk = 0, background = 0;  // compiled now / in memory / loaded / still compiling
  double compileMs = 0;
  std::vector<int> ids;
};
void jitBeginQuery(const std::string& sql);
JitQueryInfo jitEndQuery(double queryMs);  // credits the query's run time to the plans it used

struct PlanCacheRow {
  int id;
  std::string kind;
  std::string origin;  // "compiled" or "disk"
  double compileMs;
  int64_t execs;
  double avgMs, lastMs;
  size_t sourceBytes;
  std::string sql;
};
std::vector<PlanCacheRow> jitPlanCache();
std::string jitSourceOf(int id);  // generated CUDA source of a cached plan ("" if unknown)
void jitClearCache();

}  // namespace gdb
