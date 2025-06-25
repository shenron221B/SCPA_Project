#include "../include/serial.h"

/**
 * @brief Computes the matrix-vector product y = Ax serially
 *        for a sparse matrix A stored in CSR format
 *
 * For each row i of matrix A:
 *   y[i] = sum of (A[i,j] * x[j]) for all j such that A[i,j] is non-zero
 *
 * @param A Pointer (const) to the CSR matrix
 * @param x Pointer (const) to the input vector x
 * @param y Pointer to the output vector y, which will be filled with the result
 */

void serial_spmv_csr(const CSRMatrix *A, const float *x, float *y) {

    // for each row of matrix A (from 0 to A -> nrows - 1)
    for (int i = 0; i < A->nrows; i++) {
        float sum = 0.0f; // accumulator for element y[i]

        // for each non-zero elements of row 'i'
        // A->IRP[i]: index (in the array JA and AS) of the first non-zero element of row 'i'
        // A->IRP[i+1]: index of the first non-zero element of the next row (i+1)
        // A->IRP[i+1]-1: index of the last non-zero element of row 'i'
        for (int k = A->IRP[i]; k < A->IRP[i + 1]; k++) {
            // A->JA[k] contain the index of column 'j' of the current non-zero element
            // A->AS[k] contain the value A[i,j] of the current non-zero element
            // x[A->JA[k]] is the element x[j] of the vector x
            sum += A->AS[k] * x[A->JA[k]]; // execute multiplication and update sum
        }
        y[i] = sum;
    }
}