#include "../include/openmp_spmv.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Performs a binary search to find the lower bound of a value in a sorted array.
 *
 * This helper function finds the first index 'i' in the sorted array 'arr'
 * such that arr[i] >= value. It is a key component for partitioning the workload
 * based on non-zero counts, as it efficiently finds which row corresponds to a
 * certain cumulative NNZ count.
 *
 * @param arr   A pointer to the sorted integer array to search in (e.g., the CSR IRP array).
 * @param n     The number of elements in the array.
 * @param value The target value to find the lower bound for.
 * @return The first index `i` where `arr[i] >= value`.
 */
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

/**
 * @brief Pre-calculates the row boundaries for each thread to balance the workload by NNZ count.
 *
 * This function partitions the matrix rows into chunks for static scheduling in OpenMP.
 * Instead of giving each thread an equal number of rows, it gives each thread an
 * approximately equal number of non-zero elements to process. This ensures a much
 * better load balance for matrices with an irregular distribution of non-zeros.
 * The calculation itself is parallelized with OpenMP to minimize overhead.
 *
 * @param A             A pointer to the input CSRMatrix structure.
 * @param num_threads   The target number of threads for the partitioning.
 * @return              A dynamically allocated integer array of size (num_threads + 1).
 *                      The caller is responsible for freeing this memory.
 *                      `thread[i]` will process rows from `bounds[i]` to `bounds[i+1] - 1`.
 */
int* prepare_csr_bounds_by_nnz(const CSRMatrix *A, int num_threads) {
    if (num_threads <= 0 || !A) return NULL;

    int* row_bounds = (int*)malloc((num_threads + 1) * sizeof(int));
    if (!row_bounds) {
        perror("Failed to allocate row_bounds");
        return NULL;
    }

    // the first thread always starts at row 0, and the last one ends at the last row
    row_bounds[0] = 0;
    row_bounds[num_threads] = A->nrows;

    // handle the edge case of a matrix with no non-zero elements
    if (A->nnz == 0) {
        for (int i = 1; i < num_threads; ++i) row_bounds[i] = 0;
        return row_bounds;
    }

    // calculate the ideal number of non-zeros per thread
    long long nnz_per_thread = (A->nnz + num_threads - 1) / num_threads;

    // parallelize the search for boundaries to speed up the preparation phase
    #pragma omp parallel for
    for (int i = 1; i < num_threads; ++i) {
        // determine the target cumulative NNZ count for the end of thread i-1
        long long target_nnz = i * nnz_per_thread;
        // if the target exceeds the total NNZ, cap it to the last row
        if (target_nnz >= A->nnz) {
            row_bounds[i] = A->nrows;
        } else {
            // use binary search on the IRP array to find which row this NNZ count falls into
            row_bounds[i] = lower_bound_search(A->IRP, A->nrows + 1, target_nnz);
        }
    }
    return row_bounds;
}

/**
 * @brief Computes the sparse matrix-vector product (y = Ax) for a matrix in CSR format using OpenMP.
 *
 * This function parallelizes the SpMV operation by partitioning the matrix rows among
 * available threads. It uses pre-computed row boundaries to ensure a balanced workload,
 * where each thread processes a chunk of rows containing a similar number of non-zero elements.
 * This is a static partitioning strategy based on workload (NNZ) rather than row count.
 *
 * @param A             Pointer to the input CSRMatrix structure (const).
 * @param x             Pointer to the input vector x (const).
 * @param y             Pointer to the output vector y, which will be filled with the result.
 * @param num_threads   The number of OpenMP threads to use for the computation. If <= 0,
 *                      OpenMP's default number of threads will be used.
 * @param row_bounds    An array of size (num_threads + 1) that defines the start and end
 *                      row indices for each thread. `thread[tid]` processes rows from
 *                      `row_bounds[tid]` to `row_bounds[tid + 1] - 1`.
 */
void openmp_spmv_csr(const CSRMatrix *A, const float *x, float *y, int num_threads, const int* row_bounds) {
    // return early if any pointer is null to prevent segmentation faults
    if (!A || !x || !y || !row_bounds) return;

    // set the number of threads for the parallel region, if specified
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    #pragma omp parallel
    {
        // each thread gets its unique ID
        int tid = omp_get_thread_num();
        // fetch the pre-calculated start and end row indices for this specific thread
        int start_row = row_bounds[tid];
        int end_row = row_bounds[tid + 1];

        // each thread iterates over its assigned chunk of rows
        for (int i = start_row; i < end_row; i++) {
            // 'sum' is private to each thread's stack
            float sum = 0.0f;
            // the inner loop computes the dot product for the i-th row
            for (int k = A->IRP[i]; k < A->IRP[i + 1]; k++) {
                sum += A->AS[k] * x[A->JA[k]];
            }
            // write the final result to the corresponding element in the output vector
            y[i] = sum;
        }
    }
}

