#include "../include/openmp_spmv.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// --- FUNZIONE HELPER PER LA RICERCA BINARIA ---
// Trova il primo indice 'i' nell'array 'arr' tale che arr[i] >= value.
static int lower_bound_search(const int* arr, int n, int value) {
    int low = 0, high = n;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] < value) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

int* prepare_csr_bounds_by_nnz(const CSRMatrix *A, int num_threads) {
    if (num_threads <= 0 || !A) return NULL;

    int* row_bounds = (int*)malloc((num_threads + 1) * sizeof(int));
    if (!row_bounds) {
        perror("Failed to allocate row_bounds");
        return NULL;
    }

    row_bounds[0] = 0;
    row_bounds[num_threads] = A->nrows;

    // Evita divisione per zero se nnz è 0
    if (A->nnz == 0) {
        for (int i = 1; i < num_threads; ++i) row_bounds[i] = 0;
        return row_bounds;
    }

    long long nnz_per_thread = (A->nnz + num_threads - 1) / num_threads;

    #pragma omp parallel for
    for (int i = 1; i < num_threads; ++i) {
        long long target_nnz = i * nnz_per_thread;
        if (target_nnz >= A->nnz) {
            row_bounds[i] = A->nrows;
        } else {
            row_bounds[i] = lower_bound_search(A->IRP, A->nrows + 1, target_nnz);
        }
    }
    return row_bounds;
}

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

void openmp_spmv_csr(const CSRMatrix *A, const float *x, float *y, int num_threads, const int* row_bounds) {
    if (!A || !x || !y || !row_bounds) return;

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start_row = row_bounds[tid];
        int end_row = row_bounds[tid + 1];

        for (int i = start_row; i < end_row; i++) {
            float sum = 0.0f;
            for (int k = A->IRP[i]; k < A->IRP[i + 1]; k++) {
                sum += A->AS[k] * x[A->JA[k]];
            }
            y[i] = sum;
        }
    }
}

