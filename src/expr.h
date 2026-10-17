// Expression bytecode: a tiny stack VM that runs identically on the CPU (point lookups,
// write-buffer DML) and inside CUDA kernels (scans, aggregations, joins). The definitions live in
// devtypes.h so the runtime-compiled kernels (jit.cu) can share them.
#pragma once
#include "common.h"
#include "devtypes.h"
