#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "mm_reader.h"
#include "csr_utils.h"
#include "serial.h"
#include "openmp_spmv.h"
#include "CUDA/include/cuda_spmv.h"
#include <cuda_runtime.h>
#include "hll_utils.h"
#include "CUDA/include/cuda_hll_spmv.h"

// #define MATRIX_FOLDER "/home/eonardo/SCPA_Project/data/" // local path
#define MATRIX_FOLDER "/data/lpompili/SCPA_Project/data/"
#define MAX_PATH 512
#define NUM_RUNS 10 // number of run
#define BENCHMARK 1 // 1 to run all tests, 0 to run only specified configuration

// execution mode: serial or parallel
typedef enum {
    MODE_UNDEFINED,
    MODE_SERIAL_CSR,
    MODE_OPENMP_CSR,
    MODE_CUDA_CSR,
    MODE_OPENMP_HLL,
    MODE_CUDA_HLL
} ExecutionMode;

// check if matrix have a specific suffix (used for .mtx file)
int endsWith(const char *str, const char *suffix) {
    if (!str || !suffix) return 0;
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    if (lensuffix > lenstr) return 0;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

// instruction program
void print_usage(const char *prog_name) {
    printf("Usage: %s <format_mode> [options]\n", prog_name);
    printf("--- If BENCHMARK=0, runs a single test configuration ---\n");
    printf("--- If BENCHMARK=1, runs a full benchmark suite for the specified mode ---\n");
    printf("\n<format_mode> can be one of:\n");
    printf("  csr_serial                - CSR, serial execution (Reference).\n");
    printf("  csr_openmp [num_threads]  - CSR, OpenMP execution.\n");
    printf("  csr_cuda   [block_size]   - CSR, CUDA execution.\n");
    printf("  hll_openmp [hack_size] [num_threads] - HLL, OpenMP execution.\n");
    printf("  hll_cuda   [hack_size] [block_size]  - HLL, CUDA execution.\n");
    printf("\nDefaults for single runs (BENCHMARK=0):\n");
    printf("  num_threads: OpenMP system default, block_size: 256, hack_size: 32.\n");
}

int main(int argc, char *argv[]) {
    ExecutionMode mode;
    int num_threads_openmp_arg = 0; // number of threads (0 = default)
    int cuda_block_size_arg = 256; // Default block size for CUDA
    int hll_hack_size_arg = 32; // Default hack_size for HLL

    // --- 1. ARGUMENTS PARSING ---
    if (argc < 2) { // at least execution mode is required
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    // determine execution mode
    const char* mode_str = argv[1];

    if (strcmp(mode_str, "csr_serial") == 0) { mode = MODE_SERIAL_CSR; }
    else if (strcmp(mode_str, "csr_openmp") == 0) { mode = MODE_OPENMP_CSR; if (argc > 2) num_threads_openmp_arg = atoi(argv[2]); }
    else if (strcmp(mode_str, "csr_cuda") == 0) { mode = MODE_CUDA_CSR; if (argc > 2) cuda_block_size_arg = atoi(argv[2]); }
    else if (strcmp(mode_str, "hll_openmp") == 0) { mode = MODE_OPENMP_HLL; if (argc > 2) hll_hack_size_arg = atoi(argv[2]); if (argc > 3) num_threads_openmp_arg = atoi(argv[3]); }
    else if (strcmp(mode_str, "hll_cuda") == 0) { mode = MODE_CUDA_HLL; if (argc > 2) hll_hack_size_arg = atoi(argv[2]); if (argc > 3) cuda_block_size_arg = atoi(argv[3]); }
    else { printf("error: invalid mode '%s'\n", mode_str); print_usage(argv[0]); return EXIT_FAILURE; }

    // parameter validation for single test (BENCHMARK=0)
    if (num_threads_openmp_arg <= 0) num_threads_openmp_arg = 0;
    if (cuda_block_size_arg <= 0 || cuda_block_size_arg > 1024 || (cuda_block_size_arg & (cuda_block_size_arg - 1)) != 0) cuda_block_size_arg = 256;
    if (hll_hack_size_arg <= 0) hll_hack_size_arg = 32;

    printf("executing in mode: %s (BENCHMARK = %d)\n", mode_str, BENCHMARK);

    // --- 2. LOOP ON MATRIX ---
    DIR *d = opendir(MATRIX_FOLDER); // try to open the matrix directory (data)
    if (!d) {
        fprintf(stderr, "failed to open data folder: %s. ", MATRIX_FOLDER);
        perror("error details");
        return EXIT_FAILURE;
    }

    struct dirent *dir_entry; // struct to memorize entry information
    while ((dir_entry = readdir(d)) != NULL) { // read every entry of opened directory
        if (!endsWith(dir_entry->d_name, ".mtx")) continue; // check if is a .mtx file

            char matrix_filepath[MAX_PATH];
            snprintf(matrix_filepath, MAX_PATH, "%s%s", MATRIX_FOLDER, dir_entry->d_name);

            // initialize variables for this matrix
            CSRMatrix matrix_csr = {0, 0, 0LL, NULL, NULL, NULL};
            HLLMatrix matrix_hll = {0, 0, 0LL, 0, 0, NULL};
            float *x_vec = NULL;
            float *y_result = NULL;

            printf("\n=============================\n");
            printf("Processing: %s\n", matrix_filepath);
            printf("=============================\n");

            // --- 3. READ AND CONVERT MATRIX ---
            matrix_csr = read_matrix_market_to_csr(matrix_filepath);
            if (matrix_csr.IRP == NULL) {
                printf("skipping matrix %s due to read error/unsupported format.\n", dir_entry->d_name);
                continue;
            }
            printf("matrix read (CSR): nrows=%d, ncols=%d, nnz=%lld\n",
                   matrix_csr.nrows, matrix_csr.ncols, matrix_csr.nnz);

            // --- 4. VECTORS ALLOCATION ---
            x_vec = (float *)malloc(matrix_csr.ncols * sizeof(float));
            y_result = (float *)malloc(matrix_csr.nrows * sizeof(float));
            if (!x_vec || !y_result) {
                /* error */
                free_csr(&matrix_csr);
                free_hll_matrix(&matrix_hll);
                continue;
            }
            for (int i = 0; i < matrix_csr.ncols; ++i) x_vec[i] = 1.0f;

            // --- 5. EXECUTION FOR SPECIFIC MODE ---
            if (mode == MODE_SERIAL_CSR) {
                // serial mode has not a "benchmark"
                double time_total = 0;
                for (int run = 0; run < NUM_RUNS; ++run) {
                    clock_t s = clock(); serial_spmv_csr(&matrix_csr, x_vec, y_result); time_total += (double)(clock()-s)/CLOCKS_PER_SEC;
                }
                double avg_t = time_total / NUM_RUNS;
                double mflops = (matrix_csr.nnz > 0 && avg_t > 1e-9) ? (2.0 * matrix_csr.nnz) / avg_t / 1e6 : 0.0;
                printf("[PERF] Format:CSR, Mode:SerialRef, Threads:-1, BlockSize:-1, HackSize:-1, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                       avg_t, mflops, matrix_csr.nnz, dir_entry->d_name);
            }
            else if (mode == MODE_OPENMP_CSR) {
                if (BENCHMARK == 0) { // single run
                    int threads_to_use = (num_threads_openmp_arg > 0) ? num_threads_openmp_arg : omp_get_max_threads();
                    double time_total = 0;
                    for (int run = 0; run < NUM_RUNS; ++run) {
                        double s = omp_get_wtime(); openmp_spmv_csr(&matrix_csr, x_vec, y_result, num_threads_openmp_arg); time_total += omp_get_wtime() - s;
                    }
                    double avg_t = time_total / NUM_RUNS;
                    double mflops = (matrix_csr.nnz > 0 && avg_t > 1e-9) ? (2.0 * matrix_csr.nnz) / avg_t / 1e6 : 0.0;
                    printf("[PERF] Format:CSR, Mode:OpenMP, Threads:%d, BlockSize:-1, HackSize:-1, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                           threads_to_use, avg_t, mflops, matrix_csr.nnz, dir_entry->d_name);
                } else { // BENCHMARK == 1: test for all different number of threads
                    printf("--- starting OpenMP CSR benchmark sweep for %s ---\n", dir_entry->d_name);
                    for (int th = 1; th <= 40; ++th) {
                        double time_total = 0;
                        for (int run = 0; run < NUM_RUNS; ++run) {
                            double s = omp_get_wtime(); openmp_spmv_csr(&matrix_csr, x_vec, y_result, th); time_total += omp_get_wtime() - s;
                        }
                        double avg_t = time_total / NUM_RUNS;
                        double mflops = (matrix_csr.nnz > 0 && avg_t > 1e-9) ? (2.0 * matrix_csr.nnz) / avg_t / 1e6 : 0.0;
                        printf("[PERF] Format:CSR, Mode:OpenMP, Threads:%d, BlockSize:-1, HackSize:-1, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                               th, avg_t, mflops, matrix_csr.nnz, dir_entry->d_name);
                    }
                }
            }
            else if (mode == MODE_CUDA_CSR) {
                if (BENCHMARK == 0) { // single run
                    double kernel_time_total = 0; bool ok = true;
                    for (int run = 0; run < NUM_RUNS; ++run) { double kt; int status = cuda_spmv_csr_wrapper(&matrix_csr, x_vec, y_result, cuda_block_size_arg, &kt); if (status != cudaSuccess) { ok = false; break; } kernel_time_total += kt; }
                    double avg_t = ok ? (kernel_time_total / NUM_RUNS) : -1.0;
                    double mflops = (ok && matrix_csr.nnz > 0 && avg_t > 1e-9) ? (2.0 * matrix_csr.nnz) / avg_t / 1e6 : -1.0;
                    printf("[PERF] Format:CSR, Mode:CUDA, Threads:-1, BlockSize:%d, HackSize:-1, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                           cuda_block_size_arg, avg_t, mflops, matrix_csr.nnz, dir_entry->d_name);
                } else { // BENCHMARK == 1: test for every block size
                    printf("--- starting CUDA CSR benchmark sweep for %s ---\n", dir_entry->d_name);
                    int block_sizes[] = {128, 256, 512, 1024};
                    for (int i = 0; i < sizeof(block_sizes)/sizeof(block_sizes[0]); ++i) {
                        int current_bs = block_sizes[i];
                        double kernel_time_total = 0; bool ok = true;
                        for (int run = 0; run < NUM_RUNS; ++run) { double kt; int status = cuda_spmv_csr_wrapper(&matrix_csr, x_vec, y_result, current_bs, &kt); if (status != cudaSuccess) { ok = false; break; } kernel_time_total += kt; }
                        double avg_t = ok ? (kernel_time_total / NUM_RUNS) : -1.0;
                        double mflops = (ok && matrix_csr.nnz > 0 && avg_t > 1e-9) ? (2.0 * matrix_csr.nnz) / avg_t / 1.0e6 : -1.0;
                        printf("[PERF] Format:CSR, Mode:CUDA, Threads:-1, BlockSize:%d, HackSize:-1, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                               current_bs, avg_t, mflops, matrix_csr.nnz, dir_entry->d_name);
                    }
                }
            }
            // --- HLL conversion (if selected execution mode use HLL) ---
            if (mode == MODE_OPENMP_HLL || mode == MODE_CUDA_HLL) {
                printf("converting matrix to HLL format (hack_size = %d)...\n", hll_hack_size_arg);
                matrix_hll = csr_to_hll(&matrix_csr, hll_hack_size_arg);

                if (matrix_hll.num_blocks < 0) {
                    fprintf(stderr, "error: failed to convert %s to HLL, skipping HLL benchmarks for this matrix.\n", dir_entry->d_name);
                } else {
                    printf("info: HLL conversion successful for %s: %d blocks.\n", dir_entry->d_name, matrix_hll.num_blocks);

                    // allocate vectors for HLL
                    float *x_vec_hll = (float *)malloc(matrix_hll.total_cols * sizeof(float));
                    float *y_result_hll = (float *)malloc(matrix_hll.total_rows * sizeof(float));
                    if(!x_vec_hll || !y_result_hll) {/*error*/}
                    for (int i = 0; i < matrix_hll.total_cols; ++i) x_vec_hll[i] = 1.0f;

                    if (mode == MODE_OPENMP_HLL) {
                        if (BENCHMARK == 0) { // single run
                            double time_hll_omp_total = 0;
                            for (int run = 0; run < NUM_RUNS; ++run) {
                                double start_omp_wtime = omp_get_wtime();
                                openmp_spmv_hll(&matrix_hll, x_vec_hll, y_result_hll, num_threads_openmp_arg);
                                double end_omp_wtime = omp_get_wtime();
                                time_hll_omp_total += (end_omp_wtime - start_omp_wtime);
                            }
                            double avg_time_openmp_hll = time_hll_omp_total / NUM_RUNS;
                            int threads_actually_used = (num_threads_openmp_arg > 0) ? num_threads_openmp_arg : omp_get_max_threads();
                            double mflops_openmp_hll = (matrix_hll.total_nnz > 0 && avg_time_openmp_hll > 1e-9) ? (2.0 * matrix_hll.total_nnz) / avg_time_openmp_hll / 1.0e6 : 0.0;
                            printf("[PERF] Format:HLL, Mode:OpenMP, Threads:%d, BlockSize:-1, HackSize:%d, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                                   threads_actually_used, hll_hack_size_arg, avg_time_openmp_hll, mflops_openmp_hll, matrix_hll.total_nnz, dir_entry->d_name);

                            // Verification vs Serial CSR
                        } else { // BENCHMARK == 1: test for all different number of threads
                            printf("--- starting OpenMP HLL benchmark sweep for %s (hack_size=%d) ---\n", dir_entry->d_name, hll_hack_size_arg);
                            for (int th = 1; th <= 40; ++th) {
                                double time_total = 0;
                                for (int run = 0; run < NUM_RUNS; ++run) {
                                    double s = omp_get_wtime(); openmp_spmv_hll(&matrix_hll, x_vec_hll, y_result_hll, th); time_total += omp_get_wtime() - s;
                                }
                                double avg_t = time_total / NUM_RUNS;
                                double mflops = (matrix_hll.total_nnz > 0 && avg_t > 1e-9) ? (2.0 * matrix_hll.total_nnz) / avg_t / 1.0e6 : 0.0;
                                printf("[PERF] Format:HLL, Mode:OpenMP, Threads:%d, BlockSize:-1, HackSize:%d, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                                       th, hll_hack_size_arg, avg_t, mflops, matrix_hll.total_nnz, dir_entry->d_name);
                            }
                        }
                    } else if (mode == MODE_CUDA_HLL) {
                        if (BENCHMARK == 0) { // single run
                            double single_kernel_time_hll_s = 0;
                            double total_kernel_cuda_hll_time_s = 0;
                            bool cuda_run_ok = true;
                            for (int run = 0; run < NUM_RUNS; ++run) {
                                int cuda_status = cuda_spmv_hll_wrapper(&matrix_hll, x_vec_hll, y_result_hll, cuda_block_size_arg, &single_kernel_time_hll_s);
                                if (cuda_status != cudaSuccess) { total_kernel_cuda_hll_time_s = -1.0; cuda_run_ok = false; break; }
                                total_kernel_cuda_hll_time_s += single_kernel_time_hll_s;
                            }
                            double avg_time_cuda_hll = cuda_run_ok ? (total_kernel_cuda_hll_time_s / NUM_RUNS) : -1.0;
                            double mflops_cuda_hll = (cuda_run_ok && matrix_hll.total_nnz > 0 && avg_time_cuda_hll > 1e-9) ? (2.0 * matrix_hll.total_nnz) / avg_time_cuda_hll / 1.0e6 : -1.0;
                            printf("[PERF] Format:HLL, Mode:CUDA, Threads:-1, BlockSize:%d, HackSize:%d, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                                   cuda_block_size_arg, hll_hack_size_arg, avg_time_cuda_hll, mflops_cuda_hll, matrix_hll.total_nnz, dir_entry->d_name);

                            // Verification vs Serial CSR
                        } else { // BENCHMARK == 1: test for every block size
                            printf("--- starting CUDA HLL benchmark sweep for %s (hack_size=%d) ---\n", dir_entry->d_name, hll_hack_size_arg);
                            int block_sizes_to_test[] = {128, 256, 512, 1024};
                            for (int i = 0; i < sizeof(block_sizes_to_test)/sizeof(block_sizes_to_test[0]); ++i) {
                                int current_bs = block_sizes_to_test[i];
                                double kernel_time_total = 0; bool ok = true;
                                for (int run = 0; run < NUM_RUNS; ++run) {
                                    double kt;
                                    int status = cuda_spmv_hll_wrapper(&matrix_hll, x_vec_hll, y_result_hll, current_bs, &kt);
                                    if (status != cudaSuccess) { ok = false; kernel_time_total = -1.0; break; }
                                    kernel_time_total += kt;
                                }
                                double avg_t = ok ? (kernel_time_total / NUM_RUNS) : -1.0;
                                double mflops = (ok && matrix_hll.total_nnz > 0 && avg_t > 1e-9) ? (2.0 * matrix_hll.total_nnz) / avg_t / 1e6 : -1.0;
                                printf("[PERF] Format:HLL, Mode:CUDA, Threads:-1, BlockSize:%d, HackSize:%d, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                                       current_bs, hll_hack_size_arg, avg_t, mflops, matrix_hll.total_nnz, dir_entry->d_name);
                            }
                        }
                    }
                    free(x_vec_hll);
                    free(y_result_hll);
                    free_hll_matrix(&matrix_hll);
                }
            }
            // --- 6. FREE RESOURCE FOR CURRENT MATRIX ---
            printf("freeing resources for %s...\n", dir_entry->d_name);
            free_csr(&matrix_csr);
            if (matrix_hll.num_blocks >= 0) { free_hll_matrix(&matrix_hll); }
            if (x_vec) free(x_vec);
            if (y_result) free(y_result);
    }
    closedir(d); // close directory
    printf("\nall benchmark suites for mode %s completed.\n", mode_str);
    return 0;
}