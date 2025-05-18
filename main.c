#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "mm_reader.h"
#include "csr_utils.h"
#include "serial.h"
#include "openmp_spmv.h"

#define MATRIX_FOLDER "/home/eonardo/SCPA_Project/data/"
#define MAX_PATH 512
#define NUM_RUNS 10 // number of run

// execution mode: serial or parallel
typedef enum {
    MODE_SERIAL,
    MODE_OPENMP
    // todo CUDA mode
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
    printf("Usage: %s <mode> [num_threads_openmp]\n", prog_name);
    printf("Modes:\n");
    printf("  serial      - Run in serial mode.\n");
    printf("  openmp      - Run in OpenMP mode.\n");
    printf("Optional arguments:\n");
    printf("  num_threads_openmp - Number of threads for OpenMP mode (default: system max).\n");
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

    // --- Parsing of arguments ---
    // at least execution mode is required
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // determine execution mode
    if (strcmp(argv[1], "serial") == 0) {
        mode = MODE_SERIAL;
    } else if (strcmp(argv[1], "openmp") == 0) {
        mode = MODE_OPENMP;
        // if execution mode is OpenMP, check if number of threads is specified
        if (argc > 2) {
            num_threads_openmp = atoi(argv[2]);
            if (num_threads_openmp <= 0) { // invalid number of threads
                printf("warning: Invalid number of threads for OpenMP, using default\n");
                num_threads_openmp = 0; // default OpneMP
            }
        }
    } else { // mode argument not specified
        printf("error: invalid mode '%s'\n", argv[1]);
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
    if (mode == MODE_OPENMP) { // if current execution mode is OpenMP, print also the number of threads
        if (num_threads_openmp > 0) printf("Using %d OpenMP threads.\n", num_threads_openmp);
        else printf("Using default number of OpenMP threads (max available: %d).\n", omp_get_max_threads());
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
                serial_spmv(&matrix_global, x_vec, y_vec_serial_ref);
                clock_t end_t = clock(); // get end CPY time
                time_s_ref_total += (double)(end_t - start_t) / CLOCKS_PER_SEC; // accumulate time (sec)
            }
            double avg_time_serial = time_s_ref_total / NUM_RUNS; // calculate average time
            printf("[Serial Ref] Execution time: %.6f seconds\n", avg_time_serial);
            calculate_and_print_performance("Serial Ref", avg_time_serial, matrix_global.nnz);

            // execution in selected mode
            if (mode == MODE_SERIAL) {
                // in this case, the result is already calculated -> just copy
                memcpy(y_vec_parallel, y_vec_serial_ref, matrix_global.nrows * sizeof(float));
                printf("serial mode selected, results are from reference run\n");
            }
            else if (mode == MODE_OPENMP) {
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

            // clear for current matrix
            free_csr(&matrix_global);
            if (x_vec) free(x_vec); x_vec = NULL;
            if (y_vec_serial_ref) free(y_vec_serial_ref); y_vec_serial_ref = NULL;
            if (y_vec_parallel) free(y_vec_parallel); y_vec_parallel = NULL;

        }
    }

    closedir(d); // close directory

    printf("\nall matrices processed.\n");
    return 0;
}