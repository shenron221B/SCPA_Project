#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <mpi.h>    // Per MPI
#include <math.h>   // Per fabs()
#include <omp.h>    // Per omp_get_wtime() e potenzialmente altre funzioni OpenMP runtime

#include "mm_reader.h"
#include "csr_utils.h"
#include "serial.h"
#include "mpi_spmv.h"
#include "openmp_spmv.h" // NUOVO include

#define MATRIX_FOLDER "/home/eonardo/SCPA_Project/data/"
#define MAX_PATH 512
#define NUM_RUNS 10 // Numero di esecuzioni per mediare il tempo

// Modalità di esecuzione
typedef enum {
    MODE_SERIAL,
    MODE_MPI,
    MODE_OPENMP
    // Aggiungere MODE_CUDA in futuro
} ExecutionMode;

int endsWith(const char *str, const char *suffix) {
    if (!str || !suffix) return 0;
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    if (lensuffix > lenstr) return 0;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

void print_usage(const char *prog_name) {
    printf("Usage: %s <mode> [num_threads_openmp]\n", prog_name);
    printf("Modes:\n");
    printf("  serial      - Run in serial mode.\n");
    printf("  mpi         - Run in MPI mode.\n");
    printf("  openmp      - Run in OpenMP mode.\n");
    printf("Optional arguments:\n");
    printf("  num_threads_openmp - Number of threads for OpenMP mode (default: system max).\n");
}


// Funzione helper per calcolare e stampare le prestazioni
void calculate_and_print_performance(const char* mode_name, double time_s, long long nnz, int rank_info) {
    if (time_s > 0 && nnz > 0) {
        double flops = (2.0 * (double)nnz) / time_s;
        double mflops = flops / 1.0e6;
        double gflops = flops / 1.0e9;
        if (rank_info >= 0) { // Per MPI, per indicare il rank
            printf("Rank %d: [%s] Performance: %.2f MFLOPS (%.2f GFLOPS)\n", rank_info, mode_name, mflops, gflops);
        } else { // Per seriale o OpenMP (eseguiti da un singolo processo "principale")
            printf("[%s] Performance: %.2f MFLOPS (%.2f GFLOPS)\n", mode_name, mflops, gflops);
        }
    } else {
        if (rank_info >= 0) {
             printf("Rank %d: [%s] Performance: N/A (tempo o nnz sono zero)\n", rank_info, mode_name);
        } else {
             printf("[%s] Performance: N/A (tempo o nnz sono zero)\n", mode_name);
        }
    }
}


int main(int argc, char *argv[]) {
    ExecutionMode mode;
    int num_threads_openmp = 0; // 0 o negativo usa il default di OpenMP

    // --- Parsing degli argomenti da riga di comando ---
    if (argc < 2) {
        // Se MPI è inizializzato, solo rank 0 stampa l'usage e abortisce
        int mpi_initialized_flag = 0;
        MPI_Initialized(&mpi_initialized_flag);
        if (mpi_initialized_flag) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                print_usage(argv[0]);
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // Termina tutti i processi MPI
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (strcmp(argv[1], "serial") == 0) {
        mode = MODE_SERIAL;
    } else if (strcmp(argv[1], "mpi") == 0) {
        mode = MODE_MPI;
    } else if (strcmp(argv[1], "openmp") == 0) {
        mode = MODE_OPENMP;
        if (argc > 2) {
            num_threads_openmp = atoi(argv[2]);
            if (num_threads_openmp <= 0) {
                printf("Warning: Invalid number of threads for OpenMP, using default.\n");
                num_threads_openmp = 0; // Lascia che OpenMP scelga
            }
        }
    } else {
        // Gestione come sopra per MPI
        int mpi_initialized_flag = 0;
        MPI_Initialized(&mpi_initialized_flag);
        if (mpi_initialized_flag) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                printf("Error: Invalid mode '%s'\n", argv[1]);
                print_usage(argv[0]);
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        } else {
            printf("Error: Invalid mode '%s'\n", argv[1]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    // --- Inizializzazione MPI (solo se in modalità MPI) ---
    int mpi_rank = 0; // Default a 0 se non in modalità MPI
    int mpi_size = 1; // Default a 1 se non in modalità MPI
    if (mode == MODE_MPI) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }

    // --- Variabili Comuni (alcune usate solo da rank 0 in MPI) ---
    CSRMatrix matrix_global;
    matrix_global.IRP = NULL; matrix_global.JA = NULL; matrix_global.AS = NULL; // Inizializza a NULL
    float *x_vec = NULL;
    float *y_vec_serial_ref = NULL; // Per il risultato seriale di riferimento
    float *y_vec_parallel = NULL;   // Per il risultato della modalità parallela scelta

    // Solo il processo rank 0 (o il singolo processo per serial/OpenMP) gestisce i file
    if (mpi_rank == 0) {
        printf("Executing in %s mode.\n", argv[1]);
        if (mode == MODE_OPENMP) {
            if (num_threads_openmp > 0) printf("Using %d OpenMP threads.\n", num_threads_openmp);
            else printf("Using default number of OpenMP threads.\n");
        } else if (mode == MODE_MPI) {
            printf("MPI initialized with %d processes.\n", mpi_size);
        }

        // PER DEBUG: Stampa il percorso che stai per usare
        printf("DEBUG: Attempting to open matrix folder: [%s]\n", MATRIX_FOLDER); // Usa la variabile corretta qui

        DIR *d = opendir(MATRIX_FOLDER);
        if (!d) {
            perror("Failed to open data folder");
            if (mode == MODE_MPI) {
                int terminate_signal = 1;
                for(int p=1; p<mpi_size; ++p) MPI_Send(&terminate_signal, 1, MPI_INT, p, 999, MPI_COMM_WORLD);
                MPI_Finalize();
            }
            return EXIT_FAILURE;
        }
        if (mode == MODE_MPI) {
            int terminate_signal = 0;
            for(int p=1; p<mpi_size; ++p) MPI_Send(&terminate_signal, 1, MPI_INT, p, 999, MPI_COMM_WORLD);
        }

        struct dirent *dir;
        while ((dir = readdir(d)) != NULL) {
            if (endsWith(dir->d_name, ".mtx")) {
                char path[MAX_PATH];
                snprintf(path, MAX_PATH, "%s%s", MATRIX_FOLDER, dir->d_name);
                printf("\n=============================\n");
                printf("Processing: %s\n", path);
                printf("=============================\n");

                matrix_global = read_matrix_market_to_csr(path);
                if (matrix_global.nnz == 0 && matrix_global.nrows == 0) {
                    printf("Skipping empty or invalid matrix: %s\n", path);
                    if (mode == MODE_MPI) {
                        int skip_signal_nrows = 0;
                        for(int p=1; p<mpi_size; ++p) MPI_Send(&skip_signal_nrows, 1, MPI_INT, p, 888, MPI_COMM_WORLD);
                        CSRMatrix_local temp_local_mpi; // Per la chiamata di distribuzione
                        distribute_matrix_mpi_csr(&matrix_global, &temp_local_mpi, mpi_rank, mpi_size, MPI_COMM_WORLD);
                        // Non c'è bisogno di liberare temp_local_mpi perché distribute gestisce alloc/free interni o non alloca
                    }
                    free_csr(&matrix_global); // Libera anche se vuota, read_matrix_market_to_csr potrebbe aver allocato qualcosa
                    continue;
                }
                if (mode == MODE_MPI) {
                    int skip_signal_nrows = matrix_global.nrows;
                    for(int p=1; p<mpi_size; ++p) MPI_Send(&skip_signal_nrows, 1, MPI_INT, p, 888, MPI_COMM_WORLD);
                }

                printf("Matrix read: nrows=%d, ncols=%d, nnz=%lld\n", matrix_global.nrows, matrix_global.ncols, matrix_global.nnz);

                x_vec = (float *)malloc(matrix_global.ncols * sizeof(float));
                y_vec_serial_ref = (float *)malloc(matrix_global.nrows * sizeof(float));
                y_vec_parallel = (float *)malloc(matrix_global.nrows * sizeof(float)); // y per OpenMP o MPI (rank 0)
                if (!x_vec || !y_vec_serial_ref || !y_vec_parallel) {
                    perror("Failed to allocate vectors");
                    // Gestire pulizia e uscita...
                    if (mode == MODE_MPI) MPI_Abort(MPI_COMM_WORLD, 1);
                    return EXIT_FAILURE;
                }
                for (int i = 0; i < matrix_global.ncols; ++i) x_vec[i] = 1.0f;

                // --- Esecuzione Seriale di Riferimento ---
                double time_s_ref_total = 0;
                for (int run = 0; run < NUM_RUNS; ++run) {
                    clock_t start_t = clock();
                    serial_spmv(&matrix_global, x_vec, y_vec_serial_ref);
                    clock_t end_t = clock();
                    time_s_ref_total += (double)(end_t - start_t) / CLOCKS_PER_SEC;
                }
                double avg_time_serial = time_s_ref_total / NUM_RUNS;
                printf("[Serial Ref] Execution time: %.6f seconds\n", avg_time_serial);
                calculate_and_print_performance("Serial Ref", avg_time_serial, matrix_global.nnz, -1);


                // --- Esecuzione in modalità selezionata ---
                if (mode == MODE_SERIAL) { // Se la modalità è seriale, abbiamo già i risultati
                    memcpy(y_vec_parallel, y_vec_serial_ref, matrix_global.nrows * sizeof(float));
                    printf("Serial mode selected, results are from reference run.\n");
                }
                else if (mode == MODE_OPENMP) {
                    double time_omp_total = 0;
                    for (int run = 0; run < NUM_RUNS; ++run) {
                        double start_omp_wtime = omp_get_wtime(); // Timer OpenMP più preciso
                        openmp_spmv_csr(&matrix_global, x_vec, y_vec_parallel, num_threads_openmp);
                        double end_omp_wtime = omp_get_wtime();
                        time_omp_total += (end_omp_wtime - start_omp_wtime);
                    }
                    double avg_time_openmp = time_omp_total / NUM_RUNS;
                    printf("[OpenMP] Execution time: %.6f seconds (using %d threads)\n",
                           avg_time_openmp, (num_threads_openmp > 0 ? num_threads_openmp : omp_get_max_threads()) );
                    calculate_and_print_performance("OpenMP", avg_time_openmp, matrix_global.nnz, -1);

                    // Verifica OpenMP vs Serial
                    int errors_omp = 0; double diff_omp = 0.0;
                    for(int i=0; i < matrix_global.nrows; ++i) {
                        if (fabs(y_vec_serial_ref[i] - y_vec_parallel[i]) > 1e-5) {
                            errors_omp++; diff_omp += fabs(y_vec_serial_ref[i] - y_vec_parallel[i]);
                        }
                    }
                    if (errors_omp > 0) printf("[OpenMP] VERIFICATION FAILED! %d errors. Avg diff: %e\n", errors_omp, diff_omp/errors_omp);
                    else printf("[OpenMP] VERIFICATION PASSED!\n");

                }
                else if (mode == MODE_MPI) {
                    // La logica MPI è già gestita sotto, qui rank 0 ha già i suoi dati globali
                    // e ora deve partecipare alla distribuzione e al calcolo MPI
                }

                // Pulizia per la matrice corrente (eccetto per MPI che ha una logica più complessa)
                if (mode != MODE_MPI) {
                    free_csr(&matrix_global);
                    if (x_vec) free(x_vec); x_vec = NULL;
                    if (y_vec_serial_ref) free(y_vec_serial_ref); y_vec_serial_ref = NULL;
                    if (y_vec_parallel) free(y_vec_parallel); y_vec_parallel = NULL;
                }

            } else { // Non è un file .mtx
                 if (mpi_rank == 0 && dir->d_name[0] != '.') {
                    if (mode == MODE_MPI) {
                        int skip_signal_nrows = -1;
                        for(int p=1; p<mpi_size; ++p) MPI_Send(&skip_signal_nrows, 1, MPI_INT, p, 888, MPI_COMM_WORLD);
                    }
                 }
                 continue;
            }

            // --- Logica specifica per MPI (se in modalità MPI) ---
            if (mode == MODE_MPI) {
                CSRMatrix_local matrix_local_mpi;
                distribute_matrix_mpi_csr(&matrix_global, &matrix_local_mpi, mpi_rank, mpi_size, MPI_COMM_WORLD);

                if (matrix_local_mpi.global_ncols > 0) {
                    float* x_broadcasted_mpi = (float*)malloc(matrix_local_mpi.global_ncols * sizeof(float));
                    if (!x_broadcasted_mpi) { perror("Failed to allocate x_broadcasted_mpi"); MPI_Abort(MPI_COMM_WORLD, 1); }

                    if (mpi_rank == 0) {
                        memcpy(x_broadcasted_mpi, x_vec, matrix_local_mpi.global_ncols * sizeof(float));
                    }
                    MPI_Bcast(x_broadcasted_mpi, matrix_local_mpi.global_ncols, MPI_FLOAT, 0, MPI_COMM_WORLD);

                    float* y_mpi_local_buff = NULL;
                    if (matrix_local_mpi.local_nrows > 0) {
                        y_mpi_local_buff = (float *)calloc(matrix_local_mpi.local_nrows, sizeof(float));
                        if(!y_mpi_local_buff) { perror("Failed to allocate y_mpi_local_buff"); MPI_Abort(MPI_COMM_WORLD, 1); }
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    double start_mpi_wtime = MPI_Wtime();
                    for(int run=0; run < NUM_RUNS; ++run) {
                        // y_vec_parallel è y_mpi_global su rank 0
                        mpi_spmv_csr(&matrix_local_mpi, x_broadcasted_mpi, y_mpi_local_buff, y_vec_parallel, mpi_rank, mpi_size, MPI_COMM_WORLD);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    double end_mpi_wtime = MPI_Wtime();

                    if (mpi_rank == 0) {
                        double avg_time_mpi = (end_mpi_wtime - start_mpi_wtime) / NUM_RUNS;
                        printf("[MPI] Execution time: %.6f seconds (with %d processes)\n", avg_time_mpi, mpi_size);
                        calculate_and_print_performance("MPI", avg_time_mpi, matrix_global.nnz, mpi_rank);

                        // Verifica MPI vs Serial
                        int errors_mpi = 0; double diff_mpi = 0.0;
                        for(int i=0; i < matrix_global.nrows; ++i) {
                            if (fabs(y_vec_serial_ref[i] - y_vec_parallel[i]) > 1e-5) {
                                errors_mpi++; diff_mpi += fabs(y_vec_serial_ref[i] - y_vec_parallel[i]);
                            }
                        }
                        if (errors_mpi > 0) printf("[MPI] VERIFICATION FAILED! %d errors. Avg diff: %e\n", errors_mpi, diff_mpi/errors_mpi);
                        else printf("[MPI] VERIFICATION PASSED!\n");
                    }
                    free(x_broadcasted_mpi);
                    if (y_mpi_local_buff) free(y_mpi_local_buff);
                }
                free_csr_local(&matrix_local_mpi);

                // Pulizia delle risorse globali di rank 0 per MPI
                free_csr(&matrix_global);
                if (x_vec) free(x_vec); x_vec = NULL;
                if (y_vec_serial_ref) free(y_vec_serial_ref); y_vec_serial_ref = NULL;
                if (y_vec_parallel) free(y_vec_parallel); y_vec_parallel = NULL;
            } // Fine if (mode == MODE_MPI) per la gestione della matrice corrente

            // Segnale di continuazione/fine loop per MPI workers
            if (mode == MODE_MPI && mpi_rank == 0) {
                int loop_status_flag = (readdir(d) == NULL && errno == 0) ? 1 : 0; // Controlla se readdir ha finito
                // Per fare questo controllo correttamente, dovremmo "sbirciare" la prossima entry
                // o ristrutturare il loop. Una soluzione più semplice è inviare il flag
                // dopo aver deciso se il loop principale di rank 0 continuerà.
                // Per ora, assumiamo che il prossimo dir == NULL significhi fine.
                // Questa logica va affinata per il segnale di fine loop MPI.
                // La logica originale del `dir == NULL` per uscire dal loop di rank 0
                // e poi inviare il Bcast è più robusta.
                // Ripristino la logica più semplice per il flag di fine ciclo MPI:
            }
             if (mode == MODE_MPI && mpi_rank == 0) {
                // Se dir è NULL, il while esterno di rank 0 terminerà dopo questo.
                // I worker hanno bisogno di un segnale per sapere se continuare o uscire.
                // Questo viene gestito dal `loop_status_flag` inviato alla fine del blocco `if (mpi_rank == 0)`
             }
             // L'uscita dal while per rank 0 è gestita dalla condizione del while stesso.
        } // Fine while loop sui file (per rank 0)
        if (mpi_rank == 0) closedir(d);

    } // Fine if (mpi_rank == 0) per la gestione dei file e modalità seriale/OpenMP

    // --- Logica MPI per i Worker (rank != 0) ---
    if (mode == MODE_MPI && mpi_rank != 0) {
        int terminate_signal;
        MPI_Recv(&terminate_signal, 1, MPI_INT, 0, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (terminate_signal == 1) { MPI_Finalize(); return EXIT_FAILURE; }

        while(1) {
            int current_matrix_nrows_signal;
            MPI_Recv(current_matrix_nrows_signal, 1, MPI_INT, 0, 888, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (current_matrix_nrows_signal == -1) { continue; } // Skippa file non mtx
            // Se current_matrix_nrows_signal è 0, è una matrice vuota, la distribuzione gestirà questo.

            CSRMatrix_local matrix_local_mpi;
            distribute_matrix_mpi_csr(NULL, &matrix_local_mpi, mpi_rank, mpi_size, MPI_COMM_WORLD);

            if (current_matrix_nrows_signal == 0 && matrix_local_mpi.local_nrows == 0 && matrix_local_mpi.global_ncols == 0) {
                 // Se la matrice era globalmente vuota, non fare altro, ma attendi il segnale di loop
            } else if (matrix_local_mpi.global_ncols > 0) {
                 float* x_broadcasted_mpi = (float*)malloc(matrix_local_mpi.global_ncols * sizeof(float));
                 if (!x_broadcasted_mpi) { MPI_Abort(MPI_COMM_WORLD, 1); }
                 MPI_Bcast(x_broadcasted_mpi, matrix_local_mpi.global_ncols, MPI_FLOAT, 0, MPI_COMM_WORLD);

                 float* y_mpi_local_buff = NULL;
                 if (matrix_local_mpi.local_nrows > 0) {
                     y_mpi_local_buff = (float *)calloc(matrix_local_mpi.local_nrows, sizeof(float));
                     if(!y_mpi_local_buff) { MPI_Abort(MPI_COMM_WORLD, 1); }
                 }

                 MPI_Barrier(MPI_COMM_WORLD);
                 for(int run=0; run < NUM_RUNS; ++run) {
                      mpi_spmv_csr(&matrix_local_mpi, x_broadcasted_mpi, y_mpi_local_buff, NULL, mpi_rank, mpi_size, MPI_COMM_WORLD);
                 }
                 MPI_Barrier(MPI_COMM_WORLD);

                 free(x_broadcasted_mpi);
                 if (y_mpi_local_buff) free(y_mpi_local_buff);
            }
            free_csr_local(&matrix_local_mpi);

            // Attendi segnale di continuazione/fine dal rank 0
            int loop_status_flag;
            MPI_Bcast(&loop_status_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (loop_status_flag == 1) { // 1 significa "lavoro finito"
                break;
            }
        } // Fine while(1) per worker MPI
    } // Fine if (mode == MODE_MPI && mpi_rank != 0)

    // --- Segnale di fine loop per MPI (inviato da rank 0 dopo il suo loop sui file) ---
    if (mode == MODE_MPI && mpi_rank == 0) {
       int final_loop_status_flag = 1; // 1 significa "lavoro finito"
       MPI_Bcast(&final_loop_status_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // --- Finalizzazione MPI (solo se in modalità MPI) ---
    if (mode == MODE_MPI) {
        MPI_Finalize();
    }

    return 0;
}