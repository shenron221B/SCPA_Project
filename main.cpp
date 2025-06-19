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
    MODE_SERIAL_HLL,
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
    printf("  hll_serial [hack_size]    - HLL format, serial execution.\n");
    printf("                              hack_size (optional): HLL hack size for partitioning.\n");
    printf("  hll_openmp [hack_size] [num_threads] - HLL format, OpenMP execution.\n");
    printf("  hll_cuda   [hack_size] [block_size]  - HLL format, CUDA execution.\n");
    printf("\nDefaults:\n");
    printf("  num_threads: system max for OpenMP.\n");
    printf("  block_size: 256 for CUDA.\n");
    printf("  hack_size: 32 for HLL.\n");
}

// calculate and print MFLOPS/GFLOPS
void calculate_and_print_performance(const char* mode_name, double time_s, long long nnz) {
    if (time_s > 0 && nnz > 0) {
        double flops = (2.0 * (double)nnz) / time_s;
        double mflops = flops / 1.0e6;
        double gflops = flops / 1.0e9;
        printf("[%s] performance: %.2f MFLOPS (%.2f GFLOPS)\n", mode_name, mflops, gflops);
    } else {
        printf("[%s] performance: N/A (time or nnz are zero)\n", mode_name);
    }
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
    } else if (strcmp(mode_str, "hll_serial") == 0) {
        mode = MODE_SERIAL_HLL;
        if (argc > 2) hll_hack_size = atoi(argv[2]);
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
    CSRMatrix matrix_global; // structure to contain the matrix
    matrix_global.IRP = NULL; matrix_global.JA = NULL; matrix_global.AS = NULL;
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
    if (mode == MODE_SERIAL_HLL || mode == MODE_OPENMP_HLL || mode == MODE_CUDA_HLL) {
        if (hll_hack_size <= 0) {
            printf("warning: invalid or no HLL hack_size, using default 32.\n");
            hll_hack_size = 32;
        }
    }

    printf("DEBUG: attempting to open matrix folder: [%s]\n", MATRIX_FOLDER);

    DIR *d = opendir(MATRIX_FOLDER); // try to open the matrix directory (data)
    if (!d) {
        fprintf(stderr, "failed to open data folder: %s. ", MATRIX_FOLDER);
        perror("error details");
        return EXIT_FAILURE;
    }

    struct dirent *dir; // struct to memorize entry information
    while ((dir = readdir(d)) != NULL) { // read every entry of opened directory
        if (endsWith(dir->d_name, ".mtx")) { // check if is a .mtx file

            char path[MAX_PATH];
            snprintf(path, MAX_PATH, "%s%s", MATRIX_FOLDER, dir->d_name);
            printf("\n=============================\n");
            printf("Processing: %s\n", path);
            printf("=============================\n");

            matrix_global = read_matrix_market_to_csr(path); // read and convert matrix
            if (matrix_global.nnz == 0 && matrix_global.nrows == 0) { // check if matrix is valid
                printf("skipping empty or invalid matrix: %s\n", path);
                free_csr(&matrix_global);
                continue;
            }

            printf("matrix read: nrows=%d, ncols=%d, nnz=%lld\n",
                   matrix_global.nrows, matrix_global.ncols, matrix_global.nnz);

            // allocation and initialization of vectors
            x_vec = (float *)malloc(matrix_global.ncols * sizeof(float));
            y_vec_serial_ref = (float *)malloc(matrix_global.nrows * sizeof(float));
            y_vec_parallel = (float *)malloc(matrix_global.nrows * sizeof(float));

            // malloc check
            if (!x_vec || !y_vec_serial_ref || !y_vec_parallel) {
                perror("failed to allocate vectors");
                if(x_vec) free(x_vec);
                if(y_vec_serial_ref) free(y_vec_serial_ref);
                if(y_vec_parallel) free(y_vec_parallel);
                free_csr(&matrix_global);
                closedir(d);
                return EXIT_FAILURE;
            }
            // initialize vector x
            for (int i = 0; i < matrix_global.ncols; ++i) x_vec[i] = 1.0f;

            // --- serial execution ---
            double time_s_ref_total = 0;
            for (int run = 0; run < NUM_RUNS; ++run) {
                clock_t start_t = clock(); // get initial CPU time
                serial_spmv_csr(&matrix_global, x_vec, y_vec_serial_ref);
                clock_t end_t = clock(); // get end CPY time
                time_s_ref_total += (double)(end_t - start_t) / CLOCKS_PER_SEC; // accumulate time (sec)
            }
            double avg_time_serial = time_s_ref_total / NUM_RUNS; // calculate average time
            printf("[Serial Ref] Execution time: %.6f seconds\n", avg_time_serial);
            calculate_and_print_performance("Serial Ref", avg_time_serial, matrix_global.nnz);

            HLLMatrix matrix_hll;
            matrix_hll.num_blocks = -1; // initialize to indicate not yet converted

            // --- HLL conversion (if selected execution mode use HLL) ---
            if (mode == MODE_SERIAL_HLL || mode == MODE_OPENMP_HLL || mode == MODE_CUDA_HLL) {
                int hack_size_param = 32; // o leggilo da riga di comando
                printf("converting matrix to HLL format (hack_size = %d)...\n", hack_size_param);
                matrix_hll = csr_to_hll(&matrix_global, hll_hack_size);
                if (matrix_hll.num_blocks < 0 || (matrix_hll.num_blocks == 0 && matrix_hll.total_rows > 0) ) {
                    fprintf(stderr, "error: failed to convert %s to HLL format.\n", path);
                    // Libera le risorse CSR e i vettori prima di continuare con la prossima matrice
                    free_csr(&matrix_global);
                    if (x_vec) free(x_vec); x_vec = NULL;
                    if (y_vec_serial_ref) free(y_vec_serial_ref); y_vec_serial_ref = NULL;
                    if (y_vec_parallel) free(y_vec_parallel); y_vec_parallel = NULL;
                    continue; // passa alla prossima matrice nel loop
                }
                printf("HLL conversion successful: %d blocks.\n", matrix_hll.num_blocks);
            }

            // execution in selected mode
            if (mode == MODE_SERIAL_CSR) {
                // in this case, the result is already calculated -> just copy
                memcpy(y_vec_parallel, y_vec_serial_ref, matrix_global.nrows * sizeof(float));
                printf("serial mode selected, results are from reference run\n");
            }
            else if (mode == MODE_OPENMP_CSR) {
                double time_omp_total = 0;
                for (int run = 0; run < NUM_RUNS; ++run) {
                    double start_omp_wtime = omp_get_wtime(); // timer OpenMP
                    openmp_spmv_csr(&matrix_global, x_vec, y_vec_parallel, num_threads_openmp);
                    double end_omp_wtime = omp_get_wtime();
                    time_omp_total += (end_omp_wtime - start_omp_wtime); // accumulate time
                }
                double avg_time_openmp = time_omp_total / NUM_RUNS; // average time
                // determine number of threads
                int threads_actually_used = num_threads_openmp > 0 ? num_threads_openmp : omp_get_max_threads();
                printf("[OpenMP] Execution time: %.6f seconds (using %d threads)\n",
                       avg_time_openmp, threads_actually_used);
                calculate_and_print_performance("OpenMP", avg_time_openmp, matrix_global.nnz);

                // check the serial result with the OpenMP result
                int errors_omp = 0; double diff_omp = 0.0;
                for(int i=0; i < matrix_global.nrows; ++i) {
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
                double total_cuda_time_s = 0;

                // create CUDA events
                cudaEventCreate(&start_event);
                cudaEventCreate(&stop_event);

                for (int run = 0; run < NUM_RUNS; ++run) {
                    // initialize y_vec_parallel to zero if its necessary
                    // memset(y_vec_parallel, 0, matrix_global.nrows * sizeof(float)); // optional

                    cudaEventRecord(start_event, 0); // register start event

                    int cuda_status = cuda_spmv_csr_wrapper(&matrix_global, x_vec, y_vec_parallel, cuda_block_size);

                    cudaEventRecord(stop_event, 0);   // register end event
                    cudaEventSynchronize(stop_event); // wait until the end event is completed

                    if (cuda_status != 0) { // cudaSuccess is 0
                        fprintf(stderr, "CUDA SpMV execution failed with error code %d\n", cuda_status);
                        total_cuda_time_s = -1.0; // notify error
                        break;
                    }
                    cudaEventElapsedTime(&cuda_elapsed_time_ms, start_event, stop_event);
                    total_cuda_time_s += cuda_elapsed_time_ms / 1000.0; // convert ms to s
                }

                // destroy CUDA events
                cudaEventDestroy(start_event);
                cudaEventDestroy(stop_event);

                if (total_cuda_time_s >= 0) {
                    double avg_time_cuda = total_cuda_time_s / NUM_RUNS;
                    printf("[CUDA] execution time: %.6f seconds (block size: %d)\n",
                           avg_time_cuda, cuda_block_size);
                    calculate_and_print_performance("CUDA", avg_time_cuda, matrix_global.nnz);

                    // verify CUDA vs serial
                    int errors_cuda = 0; double diff_cuda = 0.0;
                    for(int i=0; i < matrix_global.nrows; ++i) {
                        if (fabs(y_vec_serial_ref[i] - y_vec_parallel[i]) > 1e-5) {
                            errors_cuda++; diff_cuda += fabs(y_vec_serial_ref[i] - y_vec_parallel[i]);
                        }
                    }
                    if (errors_cuda > 0) printf("[CUDA] VERIFICATION FAILED! %d errors. avg diff: %e\n", errors_cuda, diff_cuda/(errors_cuda == 0 ? 1 : errors_cuda));
                    else printf("[CUDA] VERIFICATION PASSED!\n");
                }
            }
            else if (mode == MODE_SERIAL_HLL) {
                double time_hll_serial_total = 0;
                for (int run = 0; run < NUM_RUNS; ++run) {
                    // Assumiamo che y_vec_parallel sia già allocato
                    // e che matrix_hll sia stata creata
                    clock_t start_t = clock();
                    serial_spmv_hll(&matrix_hll, x_vec, y_vec_parallel);
                    clock_t end_t = clock();
                    time_hll_serial_total += (double)(end_t - start_t) / CLOCKS_PER_SEC;
                }
                double avg_time_hll_serial = time_hll_serial_total / NUM_RUNS;
                printf("[Serial HLL] execution time: %.6f seconds\n", avg_time_hll_serial);
                calculate_and_print_performance("Serial_HLL", avg_time_hll_serial, matrix_hll.total_nnz); // Usa nnz originale
                // Verifica vs y_vec_serial_ref (che è CSR)
                // ... (codice di verifica) ...
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
                    double avg_time_hll_omp = time_hll_omp_total / NUM_RUNS;
                    int threads_used_omp = (num_threads_openmp > 0) ? num_threads_openmp : omp_get_max_threads();
                    printf("[OpenMP HLL] execution time: %.6f seconds (using %d threads, hack_size %d)\n",
                           avg_time_hll_omp, threads_used_omp, hll_hack_size);
                    calculate_and_print_performance("OpenMP_HLL", avg_time_hll_omp, matrix_hll.total_nnz);

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
                    printf("[OpenMP HLL] skipped due to HLL conversion failure.\n");
                }
            }
            else if (mode == MODE_CUDA_HLL) {
                if (matrix_hll.num_blocks >= 0) { // ensure HLL conversion was successful
                    cudaEvent_t start_event_hll, stop_event_hll;
                    float cuda_elapsed_time_ms_hll = 0;
                    double total_cuda_time_s_hll = 0;

                    cudaEventCreate(&start_event_hll);
                    cudaEventCreate(&stop_event_hll);

                    for (int run = 0; run < NUM_RUNS; ++run) {
                        cudaEventRecord(start_event_hll, 0);
                        int cuda_status = cuda_spmv_hll_wrapper(&matrix_hll, x_vec, y_vec_parallel, cuda_block_size);
                        cudaEventRecord(stop_event_hll, 0);
                        cudaEventSynchronize(stop_event_hll);

                        if (cuda_status != 0) {
                            fprintf(stderr, "CUDA HLL SpMV execution failed with error code %d\n", cuda_status);
                            total_cuda_time_s_hll = -1.0;
                            break;
                        }
                        cudaEventElapsedTime(&cuda_elapsed_time_ms_hll, start_event_hll, stop_event_hll);
                        total_cuda_time_s_hll += cuda_elapsed_time_ms_hll / 1000.0;
                    }

                    cudaEventDestroy(start_event_hll);
                    cudaEventDestroy(stop_event_hll);

                    if (total_cuda_time_s_hll >= 0) {
                        double avg_time_cuda_hll = total_cuda_time_s_hll / NUM_RUNS;
                        printf("[CUDA HLL] execution time: %.6f seconds (block_size: %d, hack_size %d)\n",
                               avg_time_cuda_hll, cuda_block_size, hll_hack_size);
                        calculate_and_print_performance("CUDA_HLL", avg_time_cuda_hll, matrix_hll.total_nnz);

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
                     printf("[CUDA HLL] skipped due to HLL conversion failure.\n");
                }
            }

            // clear for current matrix
            free_csr(&matrix_global);
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