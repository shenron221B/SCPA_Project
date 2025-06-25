#include "../include/openmp_spmv.h"
#include <omp.h>
#include <stdio.h>

/**
 * @brief Compute the matrix-vector product y = Ax for a sparse matrix in CSR format
 *        using OpenMP for row-wise loop parallelization.
 *
 * Implementation details:
 * - Parallelization occurs on the outer loop that iterates over the rows of matrix A.
 *   This is a common approach for SpMV because the computations for each row of y are independent.
 * - Each OpenMP thread handles a subset of rows. The distribution of rows
 *   to threads is handled by the OpenMP directive and the 'schedule' clause.
 * - The 'sum' variable (used to accumulate the dot product for a single row y[i])
 *   is declared private ('private(sum)') for each thread inside the parallel region.
 *   This is crucial to avoid race conditions, where multiple threads try to modify
 *   the same 'sum' variable at the same time, leading to incorrect results.
 * - A 'static' schedule ('schedule(static)') is used to distribute iterations (rows)
 *   to threads. With 'static', the N rows are divided into blocks of size about (N / number_threads)
 *   and each block is assigned to a thread at the beginning of the loop. This schedule has low overhead.
 *   For matrices where the number of non-zeros per row varies widely (so-called "unbalanced" matrices),
 *   other programs such as 'dynamic' or 'driven' may offer better load balancing
 *   at the cost of more overhead, but 'static' is often a good place to start.
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
    // check consistency of dimension for x and y
    // if (A->ncols <= 0 || A->nrows <= 0) {
    //     fprintf(stderr, "Errore [openmp_spmv_csr]: dimensioni della matrice non valide.\n");
    //     return;
    // }


    // set the number of threads to use, if is specified and valid
    // if num_threads is <= 0, OpenMP use the default number
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    #pragma omp parallel for shared(A, x, y) schedule(static)
    for (int i = 0; i < A->nrows; i++) {
        float sum = 0.0f; // 'sum' is private for each thread

        // the second 'for' calculate the product for row 'i' (y_i = A_i * x)
        // A->IRP[i]: index of JA and AS of the first non-zero element of the row 'i'
        // A->IRP[i+1]-1: index of the last non-zero element of row 'i'
        for (int k = A->IRP[i]; k < A->IRP[i + 1]; k++) {
            // A->JA[k]: column index of the current non-zero element
            // A->AS[k]: value of current non-zero element
            // x[A->JA[k]]: corresponding element of the vector x
            sum += A->AS[k] * x[A->JA[k]]; // accumulate product
        }
        y[i] = sum; // each thread write on its element y[i]
    }
    // at the end of #pragma construct, there is an implicit barrier: all threads wait that the others has completed
}