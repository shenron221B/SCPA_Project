#ifndef SERIAL_H
#define SERIAL_H

#include "mm_reader.h"

void serial_spmv(const CSRMatrix *A, const float *x, float *y);

#endif
