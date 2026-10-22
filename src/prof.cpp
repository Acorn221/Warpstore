// CPU sampling profiler for development: WARPSTORE_PROFILE=<file> samples the stacks of whichever
// threads use CPU (SIGPROF every ms of process CPU time) and rewrites <file> every second with the
// stacks and their counts: one line per stack, "count obj+offset obj+offset ..." (leaf first).
// Symbolize offline: scripts/profsym.py <file> (nearest symbol from nm; no debug info needed).
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/time.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int kDepth = 24;
constexpr size_t kCap = 1 << 20;
struct Sample {
  void* pc[kDepth];
  int n;
};
Sample* g_buf = nullptr;
std::atomic<size_t> g_next{0};

void onProf(int) {
  const int saved = errno;
  const size_t i = g_next.fetch_add(1, std::memory_order_relaxed);
  if (i < kCap) g_buf[i].n = backtrace(g_buf[i].pc, kDepth);
  errno = saved;
}

void dump(const char* path) {
  const size_t n = std::min(g_next.load(), kCap);
  std::map<std::vector<void*>, size_t> agg;
  for (size_t i = 0; i < n; ++i) {
    const Sample& s = g_buf[i];
    if (s.n <= 2) continue;
    agg[std::vector<void*>(s.pc + 2, s.pc + s.n)]++;  // (frames 0-1: the handler and the signal trampoline)
  }
  std::map<void*, std::string> name;
  auto where = [&](void* p) -> const std::string& {
    auto it = name.find(p);
    if (it != name.end()) return it->second;
    Dl_info di;
    char b[512];
    if (dladdr(p, &di) && di.dli_fname) {
      const char* f = strrchr(di.dli_fname, '/');
      snprintf(b, sizeof b, "%s+0x%lx", f ? f + 1 : di.dli_fname, (unsigned long)((char*)p - (char*)di.dli_fbase - 1));
    } else {
      snprintf(b, sizeof b, "?+%p", p);
    }
    return name[p] = b;
  };
  const std::string tmp = std::string(path) + ".tmp";
  FILE* f = fopen(tmp.c_str(), "w");
  if (!f) return;
  for (auto& [st, c] : agg) {
    fprintf(f, "%zu", c);
    for (void* p : st) fprintf(f, " %s", where(p).c_str());
    fputc('\n', f);
  }
  fclose(f);
  rename(tmp.c_str(), path);
}

struct Init {
  Init() {
    const char* path = getenv("WARPSTORE_PROFILE");
    if (!path || !*path) return;
    g_buf = (Sample*)calloc(kCap, sizeof(Sample));
    void* warm[4];
    backtrace(warm, 4);  // (loads the unwinder outside the signal handler)
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = onProf;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGPROF, &sa, nullptr);
    itimerval it;
    it.it_interval = {0, 1000};
    it.it_value = {0, 1000};
    setitimer(ITIMER_PROF, &it, nullptr);
    std::string p = path;
    std::thread([p] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        dump(p.c_str());
      }
    }).detach();
  }
} g_init;
}  // namespace
