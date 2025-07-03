#include "../include/openmp_spmv.h"
#include <omp.h>
#include <stdio.h>

/**
 * @brief Compute the matrix-vector product y = Ax for a sparse matrix in CSR format
 *        using OpenMP for row-wise loop parallelization.
 *
 * @param A Pointer to the CSR matrix (CSRMatrix). The matrix is unchanged (const).
 * @param x Pointer to the input vector x. It is unchanged (const).
 * @param y Pointer to the output vector y. It will be filled with the result of the product Axe.
 * @param num_threads Number of OpenMP threads the user wishes to use for the computation.
 *                    If num_threads <= 0, OpenMP will use the default number of threads
 *                    (often determined by the system, such as the number of available cores).
 */

void openmp_spmv_csr(const CSRMatrix *A, const float *x, float *y, int num_threads) {
    // check if pointer are not null
    if (!A || !A->IRP || !A->JA || !A->AS || !x || !y) {
        fprintf(stderr, "Error [openmp_spmv_csr]: one or more argument are NULL.\n");
        return;
    }

    // each thread calculate the region of row for work
    #pragma omp parallel
    {
        // each thread obtain its ID and total number of threads
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // calculate range of row for this thread
        int rows_per_thread = (A->nrows + nthreads - 1) / nthreads;
        int start_row = tid * rows_per_thread;
        int end_row = (tid + 1) * rows_per_thread;

        if (end_row > A->nrows) {
            end_row = A->nrows;
        }

        // each thread run loop on its chunk of row
        for (int i = start_row; i < end_row; i++) {
            float sum = 0.0f;
            int row_end = A->IRP[i + 1];
            for (int k = A->IRP[i]; k < row_end; k++) {
                sum += A->AS[k] * x[A->JA[k]];
            }
            y[i] = sum;
        }
    }
    // at the end of #pragma construct, there is an implicit barrier: all threads wait that the others has completed
}