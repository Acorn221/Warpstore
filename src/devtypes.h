// Plain data types and device helpers shared by the host, the ahead-of-time CUDA kernels and the
// kernels compiled at runtime with NVRTC (see jit.cu). Must stay free of host-only headers.
#pragma once
#ifdef __CUDACC_RTC__
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#else
#include <cstdint>
#include <cstring>
#endif

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

namespace gdb {

// ---- NULL: NaN in the VM; sentinels in integer storage (doubles store NaN directly).
constexpr int32_t kNullI32 = (int32_t)0x80000000;
constexpr int64_t kNullI64 = (int64_t)0x8000000000000000ull;
HD double nullv() {
  union {
    uint64_t u;
    double d;
  } x;
  x.u = 0x7ff8000000000000ull;
  return x.d;
}
HD bool isNullD(double v) { return v != v; }
HD bool truthy(double v) { return v == v && v != 0; }  // SQL: NULL and 0 are not true

// ---- Dates (Howard Hinnant's civil algorithms), usable on device ----
HD int64_t daysFromCivil(int64_t y, unsigned m, unsigned d) {
  y -= m <= 2;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = (unsigned)(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + (int64_t)doe - 719468;
}
HD void civilFromDays(int64_t z, int64_t& y, unsigned& m, unsigned& d) {
  z += 719468;
  const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = (unsigned)(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  y = (int64_t)yoe + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  d = doy - (153 * mp + 2) / 5 + 1;
  m = mp + (mp < 10 ? 3 : -9);
  y += (m <= 2);
}

// ---- Expression bytecode: a tiny stack VM. The interpreter below runs on the CPU and in the
// ahead-of-time kernels; jit.cu translates the same bytecode into typed CUDA C++.
enum Op : uint8_t {
  OP_COL,    // push column[arg][row]
  OP_CONST,  // push consts[arg]
  OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD, OP_NEG,
  OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE,
  OP_AND, OP_OR, OP_NOT,
  OP_INSET,  // pop dictionary code, push bitmap sets[arg] membership (strings: LIKE / IN / ranges)
  OP_SELECT, // pop else, then, cond -> cond ? then : else   (CASE WHEN)
  OP_YEAR,   // days since epoch -> year
  OP_ISNULL,    // pop x, push x IS NULL (0/1)
  OP_COALESCE,  // pop b, a -> a IS NULL ? b : a
  OP_NULLIF,    // pop b, a -> a = b ? NULL : a
  OP_CALL1,     // pop x -> fn[arg](x)          (scalar functions, see gdbCall1)
  OP_CALL2,     // pop b, a -> fn[arg](a, b)    (see gdbCall2)
  OP_DICTMAP,   // pop dictionary code, push maps[arg][code] (functions evaluated per dictionary entry)
  OP_INSORTED,  // pop x, push x IN maps[arg] (sorted values: IN (SELECT ..))
  OP_VDOT,      // push DOT_PRODUCT(column[arg & 255][row], maps[arg >> 8])        (VECTOR columns)
  OP_VEUC,      // push EUCLIDEAN_DISTANCE(column[arg & 255][row], maps[arg >> 8])
  OP_COLRAW,    // push the raw stored integer of column[arg] (DECIMAL sums accumulate it exactly)
};

struct Instr {
  uint8_t op;
  uint8_t isInt;  // OP_CONST: the literal is an integer / date / dictionary code (typed codegen)
  uint16_t arg;
};

struct ColView {
  const void* p;
  uint8_t t;  // 0 = I32, 1 = I64, 2 = F64 (NULL = NaN), 3 = I32 with NULLs, 4 = I64 with NULLs, 5 = float32 vectors,
              // 6 = DECIMAL (int64 scaled by 10^scale, NULL sentinel)
  uint8_t scale = 0;
  // Compressed GPU segment columns (see Table::encodeColumn): p points at a 16-byte header (int64
  // base) followed by the data. 1: uint8 + base, 2: uint16 + base, 3: uint32 + base (the top code
  // is NULL), 4: F64 dictionary (256 doubles, then uint8 codes), 5 / 6: decimal-valued F64 as uint16 /
  // uint32 + base, / 10^s. off: first row (range scans).
  uint8_t enc = 0;
  int32_t off = 0;
};
constexpr int kEncHeader = 16;
// The int64 stored bit-for-bit in a double slot (exact integer constants and GROUP BY keys).
HD long long bitsOf(double d) {
#ifdef __CUDA_ARCH__
  return __double_as_longlong(d);
#else
  long long v;
  memcpy(&v, &d, 8);
  return v;
#endif
}
HD double asBits(long long v) {
#ifdef __CUDA_ARCH__
  return __longlong_as_double(v);
#else
  double d;
  memcpy(&d, &v, 8);
  return d;
#endif
}
HD int encWidth(uint8_t enc) { return enc == 1 || enc == 4 ? 1 : enc == 2 || enc == 5 ? 2 : 4; }
// Stored integer of an encoded integer column (the type's NULL sentinel for NULL).
HD long long encIntB(const ColView& c, long long base, int64_t r) {
  const unsigned char* d = (const unsigned char*)c.p + kEncHeader;
  const int64_t i = r + c.off;
  const long long nul = c.t == 0 || c.t == 3 ? (long long)kNullI32 : (long long)kNullI64;
  if (c.enc == 1) { const unsigned v = d[i]; return v == 0xFFu ? nul : base + v; }
  if (c.enc == 2) { const unsigned v = ((const unsigned short*)d)[i]; return v == 0xFFFFu ? nul : base + v; }
  const unsigned v = ((const unsigned*)d)[i];
  return v == 0xFFFFFFFFu ? nul : base + (long long)v;
}
HD long long encInt(const ColView& c, int64_t r) { return encIntB(c, *(const long long*)c.p, r); }
HD double encDbl(const ColView& c, int64_t r) {
  const unsigned char* d = (const unsigned char*)c.p + kEncHeader;
  if (c.enc == 4) return ((const double*)d)[d[256 * 8 + r + c.off]];
  // 5 / 6: decimal-valued doubles as uint16 / uint32 offsets from an int64 base, divided by 10^s
  // (header: base, then the double 10^s); the top code is NULL.
  const long long base = *(const long long*)c.p;
  const double p = *(const double*)((const unsigned char*)c.p + 8);
  const int64_t i = r + c.off;
  if (c.enc == 5) { const unsigned v = ((const unsigned short*)d)[i]; return v == 0xFFFFu ? asBits(0x7ff8000000000000LL) : (double)(base + v) / p; }
  const unsigned v = ((const unsigned*)d)[i];
  return v == 0xFFFFFFFFu ? asBits(0x7ff8000000000000LL) : (double)(base + (long long)v) / p;
}
// Stored value of any integer-typed column view (raw or encoded).
HD long long colInt(const ColView& c, int64_t r) {
  if (c.enc) return encInt(c, r);
  return c.t == 0 || c.t == 3 ? (long long)((const int32_t*)c.p)[r] : (long long)((const int64_t*)c.p)[r];
}
HD double colDbl(const ColView& c, int64_t r) { return c.enc ? encDbl(c, r) : ((const double*)c.p)[r]; }
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
// Out-of-line loads for generated code with several compressed columns: an inlined raw-or-encoded
// branch per column makes the compiler version the loop for every combination.
static __device__ __noinline__ long long colIntNI(const ColView c, long long r) { return colInt(c, r); }
static __device__ __noinline__ double colDblNI(const ColView c, long long r) { return colDbl(c, r); }
#endif
HD double nullIfI32(long long v) { return v == kNullI32 ? nullv() : (double)v; }
HD double nullIfI64(long long v) { return v == kNullI64 ? nullv() : (double)v; }
HD double decOf(long long v, double p10) { return v == kNullI64 ? nullv() : (double)v / p10; }
HD double pow10i(int s) {
  double p = 1;
  for (int i = 0; i < s; ++i) p *= 10;
  return p;
}

constexpr int kMaxStack = 16;
constexpr int kMaxCols = 128;
constexpr int kMaxCode = 320;
constexpr int kMaxConsts = 96;
constexpr int kMaxSets = 16;  // (string predicate bitmaps and dictionary maps per compiled expression set)
constexpr int kMaxKeys = 8;
constexpr int kMaxAggs = 64;
// The interpreter (ahead-of-time kernels) handles plans up to these sizes; bigger plans are compiled.
constexpr int kInterpKeys = 4;
constexpr int kInterpAggs = 16;

struct Prog {
  int16_t start = 0, len = 0;
};

// Everything a kernel needs to evaluate a query's expressions. Passed by value as a kernel
// parameter, so it lives in the constant bank and is broadcast to all threads.
struct ExprEnv {
  Instr code[kMaxCode];
  double consts[kMaxConsts];
  const uint32_t* sets[kMaxSets];
  int32_t setSize[kMaxSets];
  const double* maps[kMaxSets];  // OP_DICTMAP tables (code -> value)
  int32_t mapSize[kMaxSets];
  ColView cols[kMaxCols];
  int* ovf = nullptr;  // device flag set by compiled int64 arithmetic that overflows (BIGINT out of range)
};

// Checked int64 arithmetic for compiled expressions: wraps like before, but flags the overflow.
HD long long iaddc(long long a, long long b, int* f) {
  const long long r = (long long)((unsigned long long)a + (unsigned long long)b);
  if (((a ^ r) & (b ^ r)) < 0 && f) *f = 1;
  return r;
}
HD long long isubc(long long a, long long b, int* f) {
  const long long r = (long long)((unsigned long long)a - (unsigned long long)b);
  if (((a ^ b) & (a ^ r)) < 0 && f) *f = 1;
  return r;
}
HD long long imulc(long long a, long long b, int* f) {
  const long long r = (long long)((unsigned long long)a * (unsigned long long)b);
#ifdef __CUDA_ARCH__
  if (__mul64hi(a, b) != (r >> 63) && f) *f = 1;
#else
  if ((__int128)a * b != (__int128)r && f) *f = 1;
#endif
  return r;
}

HD double loadCol(const ColView& c, int64_t r) {
  if (c.enc) {
    if (c.t == 2) return encDbl(c, r);
    const long long v = encInt(c, r);
    switch (c.t) {
      case 3: return v == kNullI32 ? nullv() : (double)v;
      case 4: return v == kNullI64 ? nullv() : (double)v;
      case 6: return v == kNullI64 ? nullv() : (double)v / pow10i(c.scale);
      default: return (double)v;
    }
  }
  switch (c.t) {
    case 0: return (double)((const int32_t*)c.p)[r];
    case 1: return (double)((const int64_t*)c.p)[r];
    case 3: {
      const int32_t v = ((const int32_t*)c.p)[r];
      return v == kNullI32 ? nullv() : (double)v;
    }
    case 4: {
      const int64_t v = ((const int64_t*)c.p)[r];
      return v == kNullI64 ? nullv() : (double)v;
    }
    case 6: {
      const int64_t v = ((const int64_t*)c.p)[r];
      return v == kNullI64 ? nullv() : (double)v / pow10i(c.scale);
    }
    default: return ((const double*)c.p)[r];
  }
}

// ---- Three-valued logic helpers (NULL = NaN), shared by the interpreter and generated code.
HD double cmp3(int op, double a, double b) {  // op: 0 =, 1 <>, 2 <, 3 <=, 4 >, 5 >=
  if (a != a || b != b) return nullv();
  bool r = op == 0 ? a == b : op == 1 ? a != b : op == 2 ? a < b : op == 3 ? a <= b : op == 4 ? a > b : a >= b;
  return r ? 1.0 : 0.0;
}
HD double and3(double a, double b) {
  if ((a == a && a == 0) || (b == b && b == 0)) return 0.0;
  if (a != a || b != b) return nullv();
  return 1.0;
}
HD double or3(double a, double b) {
  if ((a == a && a != 0) || (b == b && b != 0)) return 1.0;
  if (a != a || b != b) return nullv();
  return 0.0;
}
HD double not3(double a) { return a != a ? a : (a == 0 ? 1.0 : 0.0); }
HD double coalesce2(double a, double b) { return a != a ? b : a; }
HD double nullif2(double a, double b) { return a == b ? nullv() : a; }
HD double divz(double a, double b) { return b == 0 ? nullv() : a / b; }  // MySQL: x / 0 = NULL
HD double insetD(const uint32_t* bits, int size, double x) {
  if (x != x) return x;
  int64_t c = (int64_t)x;
  return (c >= 0 && c < size) ? (double)((bits[c >> 5] >> (c & 31)) & 1u) : 0.0;
}
// Vector similarity over a float32 row of `dim` elements against a constant query vector.
HD double vdotD(const float* v, int64_t row, const double* q, int dim) {
  const float* x = v + row * (int64_t)dim;
  if (x[0] != x[0]) return nullv();  // NULL vector
  float acc = 0;
  for (int i = 0; i < dim; ++i) acc += x[i] * (float)q[i];
  return acc;
}
HD double veucD(const float* v, int64_t row, const double* q, int dim) {
  const float* x = v + row * (int64_t)dim;
  if (x[0] != x[0]) return nullv();
  float acc = 0;
  for (int i = 0; i < dim; ++i) {
    float d = x[i] - (float)q[i];
    acc += d * d;
  }
  return sqrt((double)acc);
}
HD double insortedD(const double* a, int n, double x) {
  if (x != x) return x;
  int lo = 0, hi = n;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (a[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo < n && a[lo] == x ? 1.0 : 0.0;
}
HD double dictmapD(const double* map, int size, double x) {
  if (x != x) return x;
  int64_t c = (int64_t)x;
  return (c >= 0 && c < size) ? map[c] : nullv();
}

// ---- Scalar functions (ids are part of the bytecode; keep the numbering stable).
enum FnId : uint16_t {
  // one argument
  FN_ABS = 0, FN_CEIL, FN_FLOOR, FN_ROUND, FN_SQRT, FN_EXP, FN_LN, FN_LOG10, FN_LOG2, FN_SIN, FN_COS, FN_TAN,
  FN_SIGN, FN_MONTH, FN_DAY, FN_DAYOFWEEK, FN_DAYOFYEAR, FN_QUARTER, FN_WEEKDAY, FN_LAST_DAY,
  FN_DT_DATE, FN_DT_HOUR, FN_DT_MINUTE, FN_DT_SECOND, FN_DATE_TO_DT, FN_DT_UNIX, FN_UNIX_DT, FN_DT_YEAR,
  FN_DT_MONTH, FN_DT_DAY, FN_ASIN, FN_ACOS, FN_ATAN, FN_DEGREES, FN_RADIANS, FN_WEEK, FN_TO_DAYS, FN_FROM_DAYS,
  FN_DT_MICRO, FN_YEARWEEK,
  // two arguments
  FN_ROUND2 = 100, FN_POW, FN_ATAN2, FN_LEAST, FN_GREATEST, FN_TRUNCATE2, FN_ADD_MONTHS, FN_DT_ADD_MONTHS,
  FN_LOG_B, FN_IDIV, FN_DATE_TRUNC_MONTH, FN_DT_DIFF_UNIT,
};

HD int64_t floorDiv(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
constexpr int64_t kUsPerDay = 86400000000ll;

HD int64_t addMonthsDays(int64_t days, int64_t n) {
  int64_t y;
  unsigned m, d;
  civilFromDays(days, y, m, d);
  int64_t months = y * 12 + (m - 1) + n;
  y = months >= 0 ? months / 12 : (months - 11) / 12;
  m = (unsigned)(months - y * 12) + 1;
  const unsigned mdays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  unsigned lim = mdays[m - 1] + (m == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0));
  return daysFromCivil(y, m, d < lim ? d : lim);
}

HD double gdbCall1(int fn, double x) {
  if (x != x) return x;
  int64_t y;
  unsigned m, d;
  switch (fn) {
    case FN_ABS: return x < 0 ? -x : x;
    case FN_CEIL: return ceil(x);
    case FN_FLOOR: return floor(x);
    case FN_ROUND: return round(x);
    case FN_SQRT: return x < 0 ? nullv() : sqrt(x);
    case FN_EXP: return exp(x);
    case FN_LN: return x <= 0 ? nullv() : log(x);
    case FN_LOG10: return x <= 0 ? nullv() : log10(x);
    case FN_LOG2: return x <= 0 ? nullv() : log2(x);
    case FN_SIN: return sin(x);
    case FN_COS: return cos(x);
    case FN_TAN: return tan(x);
    case FN_ASIN: return asin(x);
    case FN_ACOS: return acos(x);
    case FN_ATAN: return atan(x);
    case FN_DEGREES: return x * 57.29577951308232;
    case FN_RADIANS: return x * 0.017453292519943295;
    case FN_SIGN: return x > 0 ? 1.0 : x < 0 ? -1.0 : 0.0;
    case FN_MONTH: civilFromDays((int64_t)x, y, m, d); return m;
    case FN_DAY: civilFromDays((int64_t)x, y, m, d); return d;
    case FN_DAYOFWEEK: return (double)(floorDiv((int64_t)x + 4, 7) * -7 + (int64_t)x + 4 + 1);  // 1 = Sunday
    case FN_WEEKDAY: return (double)(((int64_t)x + 3) - floorDiv((int64_t)x + 3, 7) * 7);         // 0 = Monday
    case FN_DAYOFYEAR: civilFromDays((int64_t)x, y, m, d); return (double)((int64_t)x - daysFromCivil(y, 1, 1) + 1);
    case FN_QUARTER: civilFromDays((int64_t)x, y, m, d); return (double)((m - 1) / 3 + 1);
    case FN_WEEK: {  // MySQL WEEK(d, 0): weeks starting Sunday, week 0 before the first Sunday
      civilFromDays((int64_t)x, y, m, d);
      int64_t jan1 = daysFromCivil(y, 1, 1);
      int64_t dowJan1 = (jan1 + 4) - floorDiv(jan1 + 4, 7) * 7;  // 0 = Sunday
      int64_t firstSun = jan1 + (7 - dowJan1) % 7;
      return (int64_t)x < firstSun ? 0.0 : (double)(((int64_t)x - firstSun) / 7 + 1);
    }
    case FN_YEARWEEK: {
      civilFromDays((int64_t)x, y, m, d);
      return (double)(y * 100) + gdbCall1(FN_WEEK, x);
    }
    case FN_LAST_DAY: {
      civilFromDays((int64_t)x, y, m, d);
      return (double)(addMonthsDays(daysFromCivil(y, m, 1), 1) - 1);
    }
    case FN_TO_DAYS: return x + 719528;
    case FN_FROM_DAYS: return x - 719528;
    case FN_DT_DATE: return (double)floorDiv((int64_t)x, kUsPerDay);
    case FN_DT_HOUR: return (double)(((int64_t)x - floorDiv((int64_t)x, kUsPerDay) * kUsPerDay) / 3600000000ll);
    case FN_DT_MINUTE: return (double)((((int64_t)x - floorDiv((int64_t)x, kUsPerDay) * kUsPerDay) / 60000000ll) % 60);
    case FN_DT_SECOND: return (double)((((int64_t)x - floorDiv((int64_t)x, kUsPerDay) * kUsPerDay) / 1000000ll) % 60);
    case FN_DT_MICRO: return (double)(((int64_t)x - floorDiv((int64_t)x, 1000000ll) * 1000000ll));
    case FN_DATE_TO_DT: return x * (double)kUsPerDay;
    case FN_DT_UNIX: return floor(x / 1e6);
    case FN_UNIX_DT: return x * 1e6;
    case FN_DT_YEAR: civilFromDays(floorDiv((int64_t)x, kUsPerDay), y, m, d); return (double)y;
    case FN_DT_MONTH: civilFromDays(floorDiv((int64_t)x, kUsPerDay), y, m, d); return m;
    case FN_DT_DAY: civilFromDays(floorDiv((int64_t)x, kUsPerDay), y, m, d); return d;
  }
  return nullv();
}

HD double gdbCall2(int fn, double a, double b) {
  if (fn == FN_LEAST || fn == FN_GREATEST) {
    if (a != a || b != b) return nullv();
    return fn == FN_LEAST ? (a < b ? a : b) : (a > b ? a : b);
  }
  if (a != a || b != b) return nullv();
  switch (fn) {
    case FN_ROUND2: {
      double p = pow(10.0, b);
      return round(a * p) / p;
    }
    case FN_TRUNCATE2: {
      double p = pow(10.0, b);
      return trunc(a * p) / p;
    }
    case FN_POW: return pow(a, b);
    case FN_ATAN2: return atan2(a, b);
    case FN_LOG_B: return (a <= 0 || b <= 0) ? nullv() : log(b) / log(a);
    case FN_IDIV: return b == 0 ? nullv() : trunc(a / b);
    case FN_ADD_MONTHS: return (double)addMonthsDays((int64_t)a, (int64_t)b);
    case FN_DT_ADD_MONTHS: {
      int64_t us = (int64_t)a, days = floorDiv(us, kUsPerDay);
      return (double)(addMonthsDays(days, (int64_t)b) * kUsPerDay + (us - days * kUsPerDay));
    }
    case FN_DATE_TRUNC_MONTH: {
      int64_t y;
      unsigned m, d;
      civilFromDays((int64_t)a, y, m, d);
      return (double)daysFromCivil(y, b > 0 ? 1 : m, 1);  // b > 0: year, else month
    }
  }
  return nullv();
}

// Interprets `pg` for one row, reading columns through `cols` (normally env.cols).
HD double evalProgC(const ExprEnv& env, const ColView* cols, Prog pg, int64_t row) {
  if (pg.len == 1) {
    const Instr in = env.code[pg.start];
    if (in.op == OP_COL) return loadCol(cols[in.arg], row);
    if (in.op == OP_CONST) return in.isInt ? (double)bitsOf(env.consts[in.arg]) : env.consts[in.arg];
  }
  double st[kMaxStack];
  int sp = 0;
  for (int pc = pg.start; pc < pg.start + pg.len; ++pc) {
    const Instr in = env.code[pc];
    switch (in.op) {
      case OP_COL: st[sp++] = loadCol(cols[in.arg], row); break;
      case OP_COLRAW: {
        const int64_t v = colInt(cols[in.arg], row);
        st[sp++] = v == kNullI64 ? nullv() : (double)v;
        break;
      }
      case OP_CONST: {  // integer constants hold their int64 bit pattern
        if (in.isInt) {
          st[sp++] = (double)bitsOf(env.consts[in.arg]);
        } else {
          st[sp++] = env.consts[in.arg];
        }
        break;
      }
      case OP_ADD: --sp; st[sp - 1] = st[sp - 1] + st[sp]; break;
      case OP_SUB: --sp; st[sp - 1] = st[sp - 1] - st[sp]; break;
      case OP_MUL: --sp; st[sp - 1] = st[sp - 1] * st[sp]; break;
      case OP_DIV: --sp; st[sp - 1] = divz(st[sp - 1], st[sp]); break;
      case OP_MOD: --sp; st[sp - 1] = fmod(st[sp - 1], st[sp]); break;
      case OP_NEG: st[sp - 1] = -st[sp - 1]; break;
      case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
        --sp; st[sp - 1] = cmp3(in.op - OP_EQ, st[sp - 1], st[sp]); break;
      case OP_AND: --sp; st[sp - 1] = and3(st[sp - 1], st[sp]); break;
      case OP_OR: --sp; st[sp - 1] = or3(st[sp - 1], st[sp]); break;
      case OP_NOT: st[sp - 1] = not3(st[sp - 1]); break;
      case OP_INSET: st[sp - 1] = insetD(env.sets[in.arg], env.setSize[in.arg], st[sp - 1]); break;
      case OP_SELECT: {
        sp -= 2;
        st[sp - 1] = truthy(st[sp - 1]) ? st[sp] : st[sp + 1];
        break;
      }
      case OP_ISNULL: st[sp - 1] = st[sp - 1] != st[sp - 1] ? 1.0 : 0.0; break;
      case OP_COALESCE: --sp; if (st[sp - 1] != st[sp - 1]) st[sp - 1] = st[sp]; break;
      case OP_NULLIF: --sp; if (st[sp - 1] == st[sp]) st[sp - 1] = nullv(); break;
      case OP_CALL1: st[sp - 1] = gdbCall1(in.arg, st[sp - 1]); break;
      case OP_CALL2: --sp; st[sp - 1] = gdbCall2(in.arg, st[sp - 1], st[sp]); break;
      case OP_DICTMAP: st[sp - 1] = dictmapD(env.maps[in.arg], env.mapSize[in.arg], st[sp - 1]); break;
      case OP_INSORTED: st[sp - 1] = insortedD(env.maps[in.arg], env.mapSize[in.arg], st[sp - 1]); break;
      case OP_VDOT: case OP_VEUC: {
        const int slot = in.arg & 255, m = in.arg >> 8;
        const float* v = (const float*)cols[slot].p;
        st[sp++] = in.op == OP_VDOT ? vdotD(v, row, env.maps[m], env.mapSize[m]) : veucD(v, row, env.maps[m], env.mapSize[m]);
        break;
      }
      case OP_YEAR: {
        if (st[sp - 1] != st[sp - 1]) break;
        int64_t y; unsigned m, d;
        civilFromDays((int64_t)st[sp - 1], y, m, d);
        st[sp - 1] = (double)y;
        break;
      }
    }
  }
  return st[0];
}

HD double evalProg(const ExprEnv& env, Prog pg, int64_t row) { return evalProgC(env, env.cols, pg, row); }

// ---- Aggregation plans

enum AggOp : uint8_t { AGG_SUM = 0, AGG_COUNT = 1, AGG_MIN = 2, AGG_MAX = 3 };

struct AggPlan {
  bool hasWhere = false;
  Prog where;
  int nkeys = 0;
  Prog keys[kMaxKeys];
  int naggs = 0;  // slot 0 is always a hidden COUNT(*)
  Prog aggs[kMaxAggs];
  uint8_t ops[kMaxAggs];
  uint8_t aggInt[kMaxAggs];  // the aggregate's argument is integer-valued (exact int64 accumulation)
  uint8_t aggBits[kMaxAggs] = {};  // GROUP BY: the slot holds the exact int64 result as bits (see ipart)
  uint8_t aggPart[kMaxAggs] = {};  // with aggBits on a SUM: 1 = sum of the low 32 bits, 2 = of the high 32
  uint8_t aggNull[kMaxAggs] = {};  // argument is a nullable int column (ColView t 3 / 4): iagg reads it raw, NULLs skipped
  uint8_t densePriv = 0;     // dense GROUP BY mode known when planning: 1 per-thread, 2 shared copies / global
  uint8_t keyInt[kMaxKeys] = {};  // integer key: stored as its exact int64 bits in the double key slots
  // Interpreted kernels: a device copy of the ExprEnv (passing the >4 KB kernel parameter by
  // reference into the out-of-line interpreter would copy it onto every thread's stack).
  const struct ExprEnv* genv = nullptr;
};

// Dense ("encoded") GROUP BY: when every key is an integer-like column with a small known range
// (dictionary codes, small ints, dates), group index = sum((key_k - kmin_k) * stride_k) indexes
// arrays directly instead of a hash table. With copies > 0 each block keeps `copies` private
// copies of the accumulators in shared memory (SingleStore's "multiple counter arrays"); with
// copies == 0 rows update the global arrays directly.
enum DenseKind : uint8_t { DK_COUNT, DK_SUM_I, DK_SUM_D, DK_MIN_I, DK_MAX_I, DK_MIN_D, DK_MAX_D };
struct DenseSpec {
  int64_t G = 0;  // number of dense groups; 0 => hash aggregation
  int copies = 0;
  int copyBytes = 0;
  int priv = 0;  // 1: every thread owns all G x naggs accumulators in shared memory ([slot][thread])
  int64_t kmin[kMaxKeys];
  int64_t range[kMaxKeys];
  int64_t stride[kMaxKeys];
  int slotOff[kMaxAggs];  // byte offset of each aggregate's array within one copy
  uint8_t kind[kMaxAggs];
};

// Many batches (segments + write buffers) processed by one kernel launch.
struct BatchSet {
  int nb = 0;
  int ncols = 0;
  const int64_t* n = nullptr;              // [nb] rows per batch
  const uint32_t* const* del = nullptr;    // [nb] deleted-row bitmaps (may be null)
  const ColView* cols = nullptr;           // [nb * ncols] column views per batch
};

}  // namespace gdb
