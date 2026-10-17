// Core types shared by host and device code.
#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <stdexcept>

#include "devtypes.h"  // HD, date helpers

namespace gdb {

// Logical (SQL) types.
enum class LType : uint8_t { INT, BIGINT, DOUBLE, DATE, VARCHAR, DATETIME, VECTOR, DECIMAL };
// Physical storage types. VARCHAR is dictionary-encoded into I32 codes, DATE is days since epoch.
// VEC: fixed-dimension float32 vectors (VECTOR(n)); the element size is 4 * n (see Table::colBytes).
enum class PType : uint8_t { I32 = 0, I64 = 1, F64 = 2, VEC = 5 };

inline PType physOf(LType t) {
  switch (t) {
    case LType::BIGINT:
    case LType::DATETIME: return PType::I64;
    case LType::VECTOR: return PType::VEC;
    case LType::DECIMAL: return PType::I64;  // scaled by 10^scale
    case LType::DOUBLE: return PType::F64;
    default: return PType::I32;
  }
}
inline int psize(PType p) { return p == PType::I32 ? 4 : 8; }
inline const char* ltypeName(LType t) {
  switch (t) {
    case LType::INT: return "INT";
    case LType::BIGINT: return "BIGINT";
    case LType::DOUBLE: return "DOUBLE";
    case LType::DATE: return "DATE";
    case LType::VARCHAR: return "VARCHAR";
    case LType::DATETIME: return "DATETIME";
    case LType::VECTOR: return "VECTOR";
    case LType::DECIMAL: return "DECIMAL";
  }
  return "?";
}

struct SqlError : std::runtime_error {
  int code;
  SqlError(const std::string& m, int c = 1064) : std::runtime_error(m), code(c) {}
};

bool parseDate(const std::string& s, int32_t& out);
std::string formatDate(int64_t days);
bool parseDateTime(const std::string& s, int64_t& us);  // 'YYYY-MM-DD[ HH:MM[:SS[.ffffff]]]' -> microseconds
std::string formatDateTime(int64_t us);

// A host-side scalar value (literals and result cells).
struct Value {
  enum Kind : uint8_t { INT, DBL, STR, DATE, DATETIME } kind = INT;
  int64_t i = 0;
  double d = 0;
  std::string s;
  int8_t scale = -1;  // DECIMAL values: digits after the point when formatted
  static Value Dec(double v, int sc) { Value x; x.kind = DBL; x.d = v; x.i = std::fabs(v) < 9e18 ? (int64_t)v : 0; x.scale = (int8_t)sc; return x; }
  static Value I(int64_t v) { Value x; x.kind = INT; x.i = v; x.d = (double)v; return x; }
  static Value D(double v) { Value x; x.kind = DBL; x.d = v; x.i = std::fabs(v) < 9e18 ? (int64_t)v : 0; return x; }
  static Value S(std::string v) { Value x; x.kind = STR; x.s = std::move(v); return x; }
  static Value Dt(int64_t v) { Value x; x.kind = DATE; x.i = v; x.d = (double)v; return x; }
  static Value Ts(int64_t us) { Value x; x.kind = DATETIME; x.i = us; x.d = (double)us; return x; }
  double num() const { return kind == DBL ? d : (double)i; }
  bool isNull() const { return kind == DBL && std::isnan(d); }
  static Value Null() { return D(NAN); }
  std::string toString() const;
};

}  // namespace gdb
