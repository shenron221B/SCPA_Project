#ifndef CSR_UTILS_H
#define CSR_UTILS_H

#include "mm_reader.h"

#ifdef __cplusplus // this header is included in a C++ file (main.cpp)
extern "C" {
#endif

void free_csr(CSRMatrix *A);
void print_vector(const float *v, int size);

#ifdef __cplusplus
}
#endif

#endif
