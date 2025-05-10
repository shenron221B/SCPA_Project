#include "../include/mm_reader.h"

void serial_spmv(const CSRMatrix *A, const float *x, float *y) {
    for (int i = 0; i < A->nrows; i++) {
        float sum = 0.0f;
        for (int j = A->IRP[i]; j < A->IRP[i + 1]; j++) {
            sum += A->AS[j] * x[A->JA[j]];
        }
        y[i] = sum;
    }
}
