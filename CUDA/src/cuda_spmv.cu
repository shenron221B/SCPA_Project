#include "../include/cuda_spmv.h"
#include <stdio.h>
#include <cuda_runtime.h>

// macro for CUDA error checking: it checks the result of a CUDA API call and prints an error if it failed
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "cuda error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return err; /* return the CUDA error code from the wrapper function */ \
    } \
}

/**
 * @brief CUDA kernel for CSR SpMV.
 * each thread in the launch grid computes one element of the output vector y (y_i = A_row_i * x).
 * the grid is typically 1D, and blocks are 1D.
 * global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
 * this global_thread_id is directly mapped to the row index.
 *
 * @param nrows total number of rows in the matrix.
 * @param d_IRP device pointer to the IRP (row pointers) array of the CSR matrix.
 * @param d_JA device pointer to the JA (column indices) array of the CSR matrix.
 * @param d_AS device pointer to the AS (values) array of the CSR matrix.
 * @param d_x device pointer to the input vector x.
 * @param d_y device pointer to the output vector y (where results are stored).
 */
__global__ void spmv_csr_kernel(int nrows,
                               const int *__restrict__ d_IRP,    // __restrict__ is a hint to the compiler
                               const int *__restrict__ d_JA,     // that these pointers do not alias,
                               const float *__restrict__ d_AS,   // potentially enabling optimizations.
                               const float *__restrict__ d_x,
                               float *__restrict__ d_y) {
    // calculate the global thread ID, which corresponds to the row index 'i'
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // check if this thread is responsible for a valid row
    // (the grid might be launched with more threads than rows if nrows is not a multiple of block_size)
    if (row_idx < nrows) {
        float sum_for_row = 0.0f; // private accumulator for this thread/row

        // get the start and end pointers for the current row in JA and AS arrays
        int row_start_ptr = d_IRP[row_idx];
        int row_end_ptr = d_IRP[row_idx + 1];

        // iterate over the non-zero elements in the current row
        for (int k = row_start_ptr; k < row_end_ptr; k++) {
            int col_idx = d_JA[k];         // get column index of the non-zero element
            float val = d_AS[k];           // get value of the non-zero element
            sum_for_row += val * d_x[col_idx]; // perform multiplication and add to sum
        }
        d_y[row_idx] = sum_for_row; // write the final sum to the corresponding element in the output vector y
    }
}


/**
 * @brief host wrapper function to manage the entire CUDA SpMV process for CSR format.
 * this function handles:
 * 1. allocation of memory on the GPU device.
 * 2. transfer of input data (matrix A_csr, vector x) from host (CPU) to device (GPU).
 * 3. configuration and launch of the CUDA kernel (spmv_csr_kernel).
 * 4. transfer of results (vector y) from device back to host.
 * 5. freeing of allocated GPU memory.
 *
 * @param h_A pointer to the host CSRMatrix structure (input).
 * @param h_x pointer to the host input vector x (input).
 * @param h_y pointer to the host output vector y (output, will be filled with SpMV result).
 * @param block_size desired number of threads per CUDA block for the kernel launch.
 * @param kernel_time_s pointer to a double to store the kernel execution time in seconds.
 * @return cudaSuccess (which is 0) on success, or a non-zero CUDA error code on failure.
 */
int cuda_spmv_csr_wrapper(const CSRMatrix *h_A, const float *h_x, float *h_y, int block_size, double *kernel_time_s) {
    // preliminary checks for host pointers and matrix dimensions

    if (!h_A || !h_x || !h_y) {
        fprintf(stderr, "error [cuda_spmv_csr_wrapper]: null host pointer(s) provided.\n");
        return cudaErrorInvalidValue; // an appropriate CUDA error code
    }
    // if matrix has no rows, or no columns but has rows, it's problematic or trivial
    if (h_A->nrows <= 0 ) { // allow ncols = 0 if nrows = 0
        if (h_A->nrows == 0) return cudaSuccess; // nothing to do for an empty matrix
        fprintf(stderr, "error [cuda_spmv_csr_wrapper]: invalid matrix dimensions (nrows <= 0).\n");
        return cudaErrorInvalidValue;
    }
     if (h_A->ncols <= 0 ) {
        fprintf(stderr, "error [cuda_spmv_csr_wrapper]: invalid matrix dimensions (ncols <= 0 with nrows > 0).\n");
        return cudaErrorInvalidValue;
    }
    // if matrix has rows but no non-zeros, the result y should be all zeros.
    if (h_A->nnz == 0) {
        for(int i=0; i < h_A->nrows; ++i) h_y[i] = 0.0f; // set host y to zero
        if (kernel_time_s) *kernel_time_s = 0.0;
        return cudaSuccess; // no GPU work needed
    }
    // check for null internal pointers if nnz > 0
    if (!h_A->IRP || !h_A->JA || !h_A->AS) {
        fprintf(stderr, "error [cuda_spmv_csr_wrapper]: null internal matrix array(s) for non-empty matrix.\n");
        return cudaErrorInvalidValue;
    }

    // device memory pointers
    int   *d_IRP_gpu;
    int   *d_JA_gpu;
    float *d_AS_gpu;
    float *d_x_gpu;
    float *d_y_gpu;

    cudaEvent_t start_event, stop_event; // event for kernel timing
    cudaError_t err; // for storing CUDA API call return codes

    // --- 1. allocate memory on the GPU device ---
    // IRP array: (nrows + 1) integers
    err = cudaMalloc((void **)&d_IRP_gpu, (h_A->nrows + 1) * sizeof(int));
    CUDA_CHECK(err); // check for allocation error
    // JA array: nnz integers
    err = cudaMalloc((void **)&d_JA_gpu, h_A->nnz * sizeof(int));
    CUDA_CHECK(err);
    // AS array: nnz floats
    err = cudaMalloc((void **)&d_AS_gpu, h_A->nnz * sizeof(float));
    CUDA_CHECK(err);
    // input vector x: ncols floats
    err = cudaMalloc((void **)&d_x_gpu, h_A->ncols * sizeof(float));
    CUDA_CHECK(err);
    // output vector y: nrows floats
    err = cudaMalloc((void **)&d_y_gpu, h_A->nrows * sizeof(float));
    CUDA_CHECK(err);

    // --- 2. copy data from host (CPU) memory to device (GPU) memory ---
    // copy IRP array
    err = cudaMemcpy(d_IRP_gpu, h_A->IRP, (h_A->nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    // copy JA array
    err = cudaMemcpy(d_JA_gpu, h_A->JA, h_A->nnz * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    // copy AS array
    err = cudaMemcpy(d_AS_gpu, h_A->AS, h_A->nnz * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    // copy input vector x
    err = cudaMemcpy(d_x_gpu, h_x, h_A->ncols * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    // --- Create CUDA events for timing kernel ---
    err = cudaEventCreate(&start_event); CUDA_CHECK(err);
    err = cudaEventCreate(&stop_event); CUDA_CHECK(err);

    // --- 3. configure and launch the CUDA kernel ---
    // validate and set block_size (threads per block)
    if (block_size <= 0 || block_size > 1024) { // typical max threads per block is 1024
        // fprintf(stderr, "[cuda_spmv_csr_wrapper] warning: invalid block_size %d, using 256.\n", block_size);
        block_size = 256; // a common default
    }
    // kernel launch configuration: 1D grid of 1D blocks
    dim3 threadsPerBlock(block_size);
    // calculate number of blocks needed to cover all rows
    // (h_A->nrows + threadsPerBlock.x - 1) / threadsPerBlock.x is integer ceiling division
    dim3 numBlocks((h_A->nrows + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // record start event
    err = cudaEventRecord(start_event, 0); CUDA_CHECK(err);

    // launch the kernel
    spmv_csr_kernel<<<numBlocks, threadsPerBlock>>>(
        h_A->nrows,
        d_IRP_gpu,
        d_JA_gpu,
        d_AS_gpu,
        d_x_gpu,
        d_y_gpu
    );

    // check for kernel launch errors
    err = cudaGetLastError(); CUDA_CHECK(err);

    // record stop event and synchronize
    err = cudaEventRecord(stop_event, 0); CUDA_CHECK(err);
    err = cudaEventSynchronize(stop_event); CUDA_CHECK(err); // waiting kernel and stop event

    // calculate elapsed time for the kernel
    float kernel_elapsed_ms = 0;
    err = cudaEventElapsedTime(&kernel_elapsed_ms, start_event, stop_event); CUDA_CHECK(err);
    if (kernel_time_s) *kernel_time_s = (double)kernel_elapsed_ms / 1000.0; // save time (sec)

    // destroy events
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // --- 4. copy result vector y from device (GPU) memory back to host (CPU) memory ---
    err = cudaMemcpy(h_y, d_y_gpu, h_A->nrows * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    // --- 5. free memory allocated on the GPU ---
    // it's good practice to free in reverse order of allocation, though not strictly required
    cudaFree(d_y_gpu);
    cudaFree(d_x_gpu);
    cudaFree(d_AS_gpu);
    cudaFree(d_JA_gpu);
    cudaFree(d_IRP_gpu);
    // checking errors for cudaFree is less common unless debugging specific memory issues

    return cudaSuccess;
}