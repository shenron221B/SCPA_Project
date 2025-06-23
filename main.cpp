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
    printf("\n<format_mode> can be one of:\n");
    printf("  csr_serial                - CSR format, serial execution.\n");
    printf("  csr_openmp [num_threads]  - CSR format, OpenMP execution.\n");
    printf("                              num_threads (optional): number of OpenMP threads.\n");
    printf("  csr_cuda   [block_size]   - CSR format, CUDA execution.\n");
    printf("                              block_size (optional): CUDA block size.\n");
    printf("                              hack_size (optional): HLL hack size for partitioning.\n");
    printf("  hll_openmp [hack_size] [num_threads] - HLL format, OpenMP execution.\n");
    printf("  hll_cuda   [hack_size] [block_size]  - HLL format, CUDA execution.\n");
    printf("\nDefaults:\n");
    printf("  num_threads: system max for OpenMP.\n");
    printf("  block_size: 256 for CUDA.\n");
    printf("  hack_size: 32 for HLL.\n");
}

int main(int argc, char *argv[]) {
    ExecutionMode mode;
    int num_threads_openmp = 0; // number of threads (0 = default)
    int cuda_block_size = 256; // Default block size for CUDA
    int hll_hack_size = 32; // Default hack_size for HLL

    // --- Parsing of arguments ---
    // at least execution mode is required
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // determine execution mode
    const char* mode_str = argv[1];

    if (strcmp(mode_str, "csr_serial") == 0) {
        mode = MODE_SERIAL_CSR;
    } else if (strcmp(mode_str, "csr_openmp") == 0) {
        mode = MODE_OPENMP_CSR;
        if (argc > 2) num_threads_openmp = atoi(argv[2]);
    } else if (strcmp(mode_str, "csr_cuda") == 0) {
        mode = MODE_CUDA_CSR;
        if (argc > 2) cuda_block_size = atoi(argv[2]);
    } else if (strcmp(mode_str, "hll_openmp") == 0) {
        mode = MODE_OPENMP_HLL;
        if (argc > 2) hll_hack_size = atoi(argv[2]);
        if (argc > 3) num_threads_openmp = atoi(argv[3]);
    } else if (strcmp(mode_str, "hll_cuda") == 0) {
        mode = MODE_CUDA_HLL;
        if (argc > 2) hll_hack_size = atoi(argv[2]);
        if (argc > 3) cuda_block_size = atoi(argv[3]);
    } else {
        printf("error: invalid mode '%s'\n", mode_str);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // common variables
    CSRMatrix matrix_global_csr; // structure to contain the matrix
    matrix_global_csr.IRP = NULL; matrix_global_csr.JA = NULL; matrix_global_csr.AS = NULL;
    float *x_vec = NULL; // pointer to vector x
    float *y_vec_serial_ref = NULL; // pointer to result vector y of serial execution
    float *y_vec_parallel = NULL; // pointer to result vector y of parallel execution (OpenMP o CUDA)

    // matrix elaboration
    printf("executing in %s mode.\n", argv[1]); // print current execution mode
    if (mode == MODE_OPENMP_CSR || mode == MODE_OPENMP_HLL) {
        if (num_threads_openmp <= 0) {
            printf("warning: invalid or no num_threads for OpenMP, using default.\n");
            num_threads_openmp = 0;
        }
    }
    if (mode == MODE_CUDA_CSR || mode == MODE_CUDA_HLL) {
        if (cuda_block_size <= 0 || cuda_block_size > 1024 || (cuda_block_size & (cuda_block_size - 1)) != 0) {
            printf("warning: invalid or no CUDA block_size, using default 256.\n");
            cuda_block_size = 256;
        }
    }
    if (mode == MODE_OPENMP_HLL || mode == MODE_CUDA_HLL) {
        if (hll_hack_size <= 0) {
            printf("warning: invalid or no HLL hack_size, using default 32.\n");
            hll_hack_size = 32;
        }
    }

    printf("executing in %s mode.\n", mode_str);
    if (mode == MODE_OPENMP_CSR || mode == MODE_OPENMP_HLL) {
        if (num_threads_openmp > 0) printf("using %d OpenMP threads.\n", num_threads_openmp);
        else printf("using default number of OpenMP threads (max available on system: %d).\n", omp_get_max_threads());
    }
    if (mode == MODE_CUDA_CSR || mode == MODE_CUDA_HLL) {
        printf("using CUDA block_size: %d.\n", cuda_block_size);
    }
    if (mode == MODE_OPENMP_HLL || mode == MODE_CUDA_HLL) {
        printf("using HLL hack_size: %d.\n", hll_hack_size);
    }

    printf("DEBUG: attempting to open matrix folder: [%s]\n", MATRIX_FOLDER);

    DIR *d = opendir(MATRIX_FOLDER); // try to open the matrix directory (data)
    if (!d) {
        fprintf(stderr, "failed to open data folder: %s. ", MATRIX_FOLDER);
        perror("error details");
        return EXIT_FAILURE;
    }

    struct dirent *dir_entry; // struct to memorize entry information
    while ((dir_entry = readdir(d)) != NULL) { // read every entry of opened directory
        if (endsWith(dir_entry->d_name, ".mtx")) { // check if is a .mtx file

            char matrix_filepath[MAX_PATH];
            snprintf(matrix_filepath, MAX_PATH, "%s%s", MATRIX_FOLDER, dir_entry->d_name);
            printf("\n=============================\n");
            printf("Processing: %s\n", matrix_filepath);
            printf("=============================\n");

            matrix_global_csr = read_matrix_market_to_csr(matrix_filepath); // read and convert matrix
            if (matrix_global_csr.nnz == 0 && matrix_global_csr.nrows == 0 && matrix_global_csr.IRP == NULL) { // check if matrix is valid
                printf("skipping empty or invalid matrix: %s\n", matrix_filepath);
                free_csr(&matrix_global_csr);
                continue;
            }
            if (matrix_global_csr.nrows > 0 && matrix_global_csr.IRP == NULL) { // error during allocation in reader
                printf("skipping matrix due to allocation error during read: %s\n", matrix_filepath);
                continue;
            }

            printf("matrix read: nrows=%d, ncols=%d, nnz=%lld\n",
                   matrix_global_csr.nrows, matrix_global_csr.ncols, matrix_global_csr.nnz);

            // allocation and initialization of vectors
            x_vec = (float *)malloc(matrix_global_csr.ncols * sizeof(float));
            y_vec_serial_ref = (float *)malloc(matrix_global_csr.nrows * sizeof(float));
            y_vec_parallel = (float *)malloc(matrix_global_csr.nrows * sizeof(float));

            // malloc check
            if (!x_vec || !y_vec_serial_ref || !y_vec_parallel) {
                perror("failed to allocate vectors");
                if(x_vec) free(x_vec);
                if(y_vec_serial_ref) free(y_vec_serial_ref);
                if(y_vec_parallel) free(y_vec_parallel);
                free_csr(&matrix_global_csr);
                closedir(d);
                return EXIT_FAILURE;
            }
            // initialize vector x
            for (int i = 0; i < matrix_global_csr.ncols; ++i) x_vec[i] = 1.0f;

            // --- serial execution ---
            double time_s_ref_total_csr = 0;
            for (int run = 0; run < NUM_RUNS; ++run) {
                clock_t start_t = clock(); // get initial CPU time
                serial_spmv_csr(&matrix_global_csr, x_vec, y_vec_serial_ref);
                clock_t end_t = clock(); // get end CPY time
                time_s_ref_total_csr += (double)(end_t - start_t) / CLOCKS_PER_SEC; // accumulate time (sec)
            }
            double avg_time_serial_csr = time_s_ref_total_csr / NUM_RUNS; // calculate average time
            double mflops_serial_csr = 0.0;
            if (matrix_global_csr.nnz > 0 && avg_time_serial_csr > 1e-9) {
                mflops_serial_csr = (2.0 * (double)matrix_global_csr.nnz) / avg_time_serial_csr / 1.0e6;
            }
            printf("[PERF] Format:CSR, Mode:SerialRef, Threads:-1, BlockSize:-1, HackSize:-1, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                   avg_time_serial_csr,
                   mflops_serial_csr,
                   matrix_global_csr.nnz,
                   dir_entry->d_name);

            HLLMatrix matrix_hll;
            matrix_hll.num_blocks = -1; // initialize to indicate not yet converted

            // --- HLL conversion (if selected execution mode use HLL) ---
            if (mode == MODE_OPENMP_HLL || mode == MODE_CUDA_HLL) {
                int hack_size_param = 32; // or read from cli
                printf("converting matrix to HLL format (hack_size = %d)...\n", hack_size_param);
                matrix_hll = csr_to_hll(&matrix_global_csr, hll_hack_size);
                if (matrix_hll.num_blocks < 0 || (matrix_hll.num_blocks == 0 && matrix_hll.total_rows > 0) ) {
                    fprintf(stderr, "error: failed to convert %s to HLL format.\n", matrix_filepath);
                    // free of CSR resources and vectors
                    free_csr(&matrix_global_csr);
                    if (x_vec) free(x_vec); x_vec = NULL;
                    if (y_vec_serial_ref) free(y_vec_serial_ref); y_vec_serial_ref = NULL;
                    if (y_vec_parallel) free(y_vec_parallel); y_vec_parallel = NULL;
                    continue; // continue with next matrix
                }
                printf("HLL conversion successful: %d blocks.\n", matrix_hll.num_blocks);
            }

            // execution in selected mode
            if (mode == MODE_SERIAL_CSR) {
                // in this case, the result is already calculated -> just copy
                memcpy(y_vec_parallel, y_vec_serial_ref, matrix_global_csr.nrows * sizeof(float));
                printf("info: serial_csr mode selected, result is from reference CSR run for %s.\n", dir_entry->d_name);
            }
            else if (mode == MODE_OPENMP_CSR) {
                double time_omp_csr_total = 0;
                for (int run = 0; run < NUM_RUNS; ++run) {
                    double start_omp_wtime = omp_get_wtime(); // timer OpenMP
                    openmp_spmv_csr(&matrix_global_csr, x_vec, y_vec_parallel, num_threads_openmp);
                    double end_omp_wtime = omp_get_wtime();
                    time_omp_csr_total += (end_omp_wtime - start_omp_wtime); // accumulate time
                }
                double avg_time_openmp_csr = time_omp_csr_total / NUM_RUNS; // average time
                // determine number of threads
                int threads_actually_used = num_threads_openmp > 0 ? num_threads_openmp : omp_get_max_threads();
                double mflops_openmp_csr = 0.0;
                if (matrix_global_csr.nnz > 0 && avg_time_openmp_csr > 1e-9) {
                    mflops_openmp_csr = (2.0 * (double)matrix_global_csr.nnz) / avg_time_openmp_csr / 1.0e6;
                }
                printf("[PERF] Format:CSR, Mode:OpenMP, Threads:%d, BlockSize:-1, HackSize:-1, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                       threads_actually_used,
                       avg_time_openmp_csr,
                       mflops_openmp_csr,
                       matrix_global_csr.nnz,
                       dir_entry->d_name);

                // check the serial result with the OpenMP result
                int errors_omp = 0; double diff_omp = 0.0;
                for(int i=0; i < matrix_global_csr.nrows; ++i) {
                    // check element by element
                    if (fabs(y_vec_serial_ref[i] - y_vec_parallel[i]) > 1e-5) {
                        errors_omp++; diff_omp += fabs(y_vec_serial_ref[i] - y_vec_parallel[i]);
                    }
                }
                if (errors_omp > 0) printf("[OpenMP] VERIFICATION FAILED! %d errors. Avg diff: %e\n", errors_omp, diff_omp/errors_omp);
                else printf("[OpenMP] VERIFICATION PASSED!\n");
            }
            else if (mode == MODE_CUDA_CSR) {
                // measure time for CUDA
                cudaEvent_t start_event, stop_event;
                float cuda_elapsed_time_ms = 0;
                double total_cuda_csr_time_s = 0;

                // create CUDA events
                cudaEventCreate(&start_event);
                cudaEventCreate(&stop_event);

                for (int run = 0; run < NUM_RUNS; ++run) {
                    // initialize y_vec_parallel to zero if its necessary
                    // memset(y_vec_parallel, 0, matrix_global.nrows * sizeof(float)); // optional

                    cudaEventRecord(start_event, 0); // register start event

                    int cuda_status = cuda_spmv_csr_wrapper(&matrix_global_csr, x_vec, y_vec_parallel, cuda_block_size);

                    cudaEventRecord(stop_event, 0);   // register end event
                    cudaEventSynchronize(stop_event); // wait until the end event is completed

                    if (cuda_status != 0) { // cudaSuccess is 0
                        fprintf(stderr, "CUDA SpMV execution failed with error code %d\n", cuda_status);
                        total_cuda_csr_time_s = -1.0; // notify error
                        break;
                    }
                    cudaEventElapsedTime(&cuda_elapsed_time_ms, start_event, stop_event);
                    total_cuda_csr_time_s += cuda_elapsed_time_ms / 1000.0; // convert ms to s
                }

                // destroy CUDA events
                cudaEventDestroy(start_event);
                cudaEventDestroy(stop_event);

                if (total_cuda_csr_time_s >= 0) {
                    double avg_time_cuda_csr = total_cuda_csr_time_s / NUM_RUNS;
                    double mflops_cuda_csr = 0.0;
                    if (matrix_global_csr.nnz > 0 && avg_time_cuda_csr > 1e-9) {
                        mflops_cuda_csr = (2.0 * (double)matrix_global_csr.nnz) / avg_time_cuda_csr / 1.0e6;
                    }
                    printf("[PERF] Format:CSR, Mode:CUDA, Threads:-1, BlockSize:%d, HackSize:-1, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                           cuda_block_size,
                           avg_time_cuda_csr,
                           mflops_cuda_csr,
                           matrix_global_csr.nnz,
                           dir_entry->d_name);
                    // verify CUDA vs serial
                    int errors_cuda = 0; double diff_cuda = 0.0;
                    for(int i=0; i < matrix_global_csr.nrows; ++i) {
                        if (fabs(y_vec_serial_ref[i] - y_vec_parallel[i]) > 1e-5) {
                            errors_cuda++; diff_cuda += fabs(y_vec_serial_ref[i] - y_vec_parallel[i]);
                        }
                    }
                    if (errors_cuda > 0) printf("[CUDA] VERIFICATION FAILED! %d errors. avg diff: %e\n", errors_cuda, diff_cuda/(errors_cuda == 0 ? 1 : errors_cuda));
                    else printf("[CUDA] VERIFICATION PASSED!\n");
                }
            }
            else if (mode == MODE_OPENMP_HLL) {
                double time_hll_omp_total = 0;
                if (matrix_hll.num_blocks >= 0) { // ensure HLL conversion was successful
                    for (int run = 0; run < NUM_RUNS; ++run) {
                        double start_omp_wtime = omp_get_wtime();
                        openmp_spmv_hll(&matrix_hll, x_vec, y_vec_parallel, num_threads_openmp);
                        double end_omp_wtime = omp_get_wtime();
                        time_hll_omp_total += (end_omp_wtime - start_omp_wtime);
                    }
                    double avg_time_openmp_hll = time_hll_omp_total / NUM_RUNS;
                    int threads_actually_used = (num_threads_openmp > 0) ? num_threads_openmp : omp_get_max_threads();
                    double mflops_openmp_hll = 0.0;
                    if (matrix_hll.total_nnz > 0 && avg_time_openmp_hll > 1e-9) {
                        mflops_openmp_hll = (2.0 * (double)matrix_hll.total_nnz) / avg_time_openmp_hll / 1.0e6;
                    }
                    printf("[PERF] Format:HLL, Mode:OpenMP, Threads:%d, BlockSize:-1, HackSize:%d, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                           threads_actually_used,
                           hll_hack_size,
                           avg_time_openmp_hll,
                           mflops_openmp_hll,
                           matrix_hll.total_nnz,
                           dir_entry->d_name);

                    // verification OpenMP HLL vs Serial CSR
                    int errors_hll_omp = 0; double diff_hll_omp = 0.0;
                    for(int i=0; i < matrix_hll.total_rows; ++i) {
                        if (fabs(y_vec_serial_ref[i] - y_vec_parallel[i]) > 1e-5) {
                            errors_hll_omp++; diff_hll_omp += fabs(y_vec_serial_ref[i] - y_vec_parallel[i]);
                        }
                    }
                    if (errors_hll_omp > 0) printf("[OpenMP HLL] VERIFICATION FAILED! %d errors. avg diff: %e\n", errors_hll_omp, diff_hll_omp/(errors_hll_omp == 0 ? 1 : errors_hll_omp));
                    else printf("[OpenMP HLL] VERIFICATION PASSED!\n");
                } else {
                    printf("info: OpenMP HLL skipped for %s due to HLL conversion issue.\n", dir_entry->d_name);

                    printf("[PERF] Format:HLL, Mode:OpenMP, Threads:%d, BlockSize:-1, HackSize:%d, Time_s:-1.00, MFLOPS:-1.00, NNZ:%lld, Matrix:%s\n",
                           (num_threads_openmp > 0 ? num_threads_openmp : omp_get_max_threads()),
                           hll_hack_size,
                           matrix_global_csr.nnz,
                           dir_entry->d_name);
                }
            }
            else if (mode == MODE_CUDA_HLL) {
                if (matrix_hll.num_blocks >= 0) { // ensure HLL conversion was successful
                    cudaEvent_t start_event_hll, stop_event_hll;
                    float cuda_elapsed_time_ms_hll = 0;
                    double total_cuda_hll_time_s = 0;

                    cudaEventCreate(&start_event_hll);
                    cudaEventCreate(&stop_event_hll);

                    for (int run = 0; run < NUM_RUNS; ++run) {
                        cudaEventRecord(start_event_hll, 0);
                        int cuda_status = cuda_spmv_hll_wrapper(&matrix_hll, x_vec, y_vec_parallel, cuda_block_size);
                        cudaEventRecord(stop_event_hll, 0);
                        cudaEventSynchronize(stop_event_hll);

                        if (cuda_status != 0) {
                            fprintf(stderr, "CUDA HLL SpMV execution failed with error code %d\n", cuda_status);
                            total_cuda_hll_time_s = -1.0;
                            break;
                        }
                        cudaEventElapsedTime(&cuda_elapsed_time_ms_hll, start_event_hll, stop_event_hll);
                        total_cuda_hll_time_s += cuda_elapsed_time_ms_hll / 1000.0;
                    }

                    cudaEventDestroy(start_event_hll);
                    cudaEventDestroy(stop_event_hll);

                    if (total_cuda_hll_time_s >= 0) {
                        double avg_time_cuda_hll = total_cuda_hll_time_s / NUM_RUNS;
                        double mflops_cuda_hll = 0.0;
                        if (matrix_hll.total_nnz > 0 && avg_time_cuda_hll > 1e-9) {
                            mflops_cuda_hll = (2.0 * (double)matrix_hll.total_nnz) / avg_time_cuda_hll / 1.0e6;
                        }
                        printf("[PERF] Format:HLL, Mode:CUDA, Threads:-1, BlockSize:%d, HackSize:%d, Time_s:%.8f, MFLOPS:%.2f, NNZ:%lld, Matrix:%s\n",
                               cuda_block_size,
                               hll_hack_size,
                               avg_time_cuda_hll,
                               mflops_cuda_hll,
                               matrix_hll.total_nnz,
                               dir_entry->d_name);

                        // verification CUDA HLL vs Serial CSR
                        int errors_cuda_hll = 0; double diff_cuda_hll = 0.0;
                        for(int i=0; i < matrix_hll.total_rows; ++i) {
                            if (fabs(y_vec_serial_ref[i] - y_vec_parallel[i]) > 1e-5) {
                                errors_cuda_hll++; diff_cuda_hll += fabs(y_vec_serial_ref[i] - y_vec_parallel[i]);
                            }
                        }
                        if (errors_cuda_hll > 0) printf("[CUDA HLL] VERIFICATION FAILED! %d errors. avg diff: %e\n", errors_cuda_hll, diff_cuda_hll/(errors_cuda_hll == 0 ? 1 : errors_cuda_hll));
                        else printf("[CUDA HLL] VERIFICATION PASSED!\n");
                    }
                } else {
                    printf("info: CUDA HLL skipped for %s due to HLL conversion issue or empty HLL matrix.\n", dir_entry->d_name);
                    printf("[PERF] Format:HLL, Mode:CUDA, Threads:-1, BlockSize:%d, HackSize:%d, Time_s:-1.00, MFLOPS:-1.00, NNZ:%lld, Matrix:%s\n",
                          cuda_block_size,
                          hll_hack_size,
                          matrix_global_csr.nnz,
                          dir_entry->d_name);
                }
            }

            // clear for current matrix
            free_csr(&matrix_global_csr);
            if (matrix_hll.num_blocks >= 0) { // free HLL only if its converted
                free_hll_matrix(&matrix_hll);
            }
            if (x_vec) free(x_vec); x_vec = NULL;
            if (y_vec_serial_ref) free(y_vec_serial_ref); y_vec_serial_ref = NULL;
            if (y_vec_parallel) free(y_vec_parallel); y_vec_parallel = NULL;
        }
    }

    closedir(d); // close directory
    printf("\nall matrices processed.\n");
    return 0;
}