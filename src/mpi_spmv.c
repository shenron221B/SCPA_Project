#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Per memcpy
#include "../include/mpi_spmv.h"
#include "../include/serial.h" // Per riutilizzare serial_spmv localmente

void free_csr_local(CSRMatrix_local *A_local) {
    if (A_local->IRP_local) free(A_local->IRP_local);
    if (A_local->JA_local) free(A_local->JA_local);
    if (A_local->AS_local) free(A_local->AS_local);
    A_local->IRP_local = NULL;
    A_local->JA_local = NULL;
    A_local->AS_local = NULL;
}

void distribute_matrix_mpi_csr(const CSRMatrix *A_global, CSRMatrix_local *A_local, int rank, int size, MPI_Comm comm) {
    A_local->IRP_local = NULL;
    A_local->JA_local = NULL;
    A_local->AS_local = NULL;

    if (rank == 0) {
        if (!A_global || A_global->nrows == 0) {
            // Segnala agli altri processi che non c'è lavoro da fare
            for (int p = 0; p < size; ++p) {
                 A_local->local_nrows = 0; // Inizializza per rank 0
                 A_local->global_ncols = 0;
                 A_local->local_nnz = 0;
                if (p > 0) {
                    int M_loc = 0, N_glob = 0, NNZ_loc = 0;
                    MPI_Send(&M_loc, 1, MPI_INT, p, 0, comm);
                    MPI_Send(&N_glob, 1, MPI_INT, p, 1, comm);
                    MPI_Send(&NNZ_loc, 1, MPI_INT, p, 2, comm);
                }
            }
            if (!A_global) printf("Rank 0: Matrice globale non fornita per la distribuzione.\n");
            else printf("Rank 0: Matrice globale vuota (nrows=0), nessuna distribuzione.\n");
            return;
        }

        int M = A_global->nrows;
        A_local->global_ncols = A_global->ncols; // Salva per rank 0

        // Calcola la suddivisione delle righe
        int *rows_per_rank = (int *)malloc(size * sizeof(int));
        int *displs_rows = (int *)malloc(size * sizeof(int));    // Indice della prima riga globale per ogni rank
        int *displs_nnz = (int *)malloc(size * sizeof(int));     // Indice del primo non-zero globale per ogni rank

        int base_rows = M / size;
        int remainder_rows = M % size;
        int current_row = 0;
        int current_nnz_offset = 0;

        for (int i = 0; i < size; i++) {
            rows_per_rank[i] = base_rows + (i < remainder_rows ? 1 : 0);
            displs_rows[i] = current_row;
            displs_nnz[i] = A_global->IRP[current_row]; // L'offset per JA/AS è dato da IRP[prima_riga_del_rank]

            if (rows_per_rank[i] == 0 && M > 0) { // Caso in cui ci sono più processi che righe
                 // Questo rank non avrà righe, ma potrebbe essere gestito meglio
                 // Per ora, se un rank non ha righe, local_nnz sarà 0.
            }
            current_row += rows_per_rank[i];
        }

        // Invia i metadati e i dati delle sottomatrici agli altri processi
        for (int p = 1; p < size; p++) {
            int M_loc = rows_per_rank[p];
            int NNZ_loc = (M_loc == 0) ? 0 : (A_global->IRP[displs_rows[p] + M_loc] - A_global->IRP[displs_rows[p]]);

            MPI_Send(&M_loc, 1, MPI_INT, p, 0, comm);
            MPI_Send(&A_global->ncols, 1, MPI_INT, p, 1, comm); // N globale
            MPI_Send(&NNZ_loc, 1, MPI_INT, p, 2, comm);

            if (M_loc > 0 && NNZ_loc > 0) {
                // Invia JA_local e AS_local
                MPI_Send(A_global->JA + displs_nnz[p], NNZ_loc, MPI_INT, p, 3, comm);
                MPI_Send(A_global->AS + displs_nnz[p], NNZ_loc, MPI_FLOAT, p, 4, comm);

                // Costruisci e invia IRP_local (ribasato)
                int *IRP_loc_temp = (int *)malloc((M_loc + 1) * sizeof(int));
                IRP_loc_temp[0] = 0;
                for (int i = 0; i < M_loc; i++) {
                    IRP_loc_temp[i + 1] = A_global->IRP[displs_rows[p] + i + 1] - A_global->IRP[displs_rows[p]];
                }
                MPI_Send(IRP_loc_temp, M_loc + 1, MPI_INT, p, 5, comm);
                free(IRP_loc_temp);
            } else if (M_loc > 0 && NNZ_loc == 0) { // Righe vuote
                 int *IRP_loc_temp = (int *)calloc((M_loc + 1), sizeof(int)); // Tutti zeri
                 MPI_Send(IRP_loc_temp, M_loc + 1, MPI_INT, p, 5, comm);
                 free(IRP_loc_temp);
            }
        }

        // Imposta i dati locali per il rank 0
        A_local->local_nrows = rows_per_rank[0];
        A_local->local_nnz = (A_local->local_nrows == 0) ? 0 : (A_global->IRP[rows_per_rank[0]] - A_global->IRP[0]);

        if (A_local->local_nrows > 0) {
            A_local->IRP_local = (int *)malloc((A_local->local_nrows + 1) * sizeof(int));
            A_local->JA_local = (int *)malloc(A_local->local_nnz * sizeof(int));
            A_local->AS_local = (float *)malloc(A_local->local_nnz * sizeof(float));

            memcpy(A_local->IRP_local, A_global->IRP, (A_local->local_nrows + 1) * sizeof(int));
            // IRP per rank 0 è già ribasato se A_global->IRP[0] è 0
            // Se A_global->IRP[0] non fosse 0 (non standard per CSR), andrebbe ribasato
            if (A_local->local_nnz > 0) {
                 memcpy(A_local->JA_local, A_global->JA, A_local->local_nnz * sizeof(int));
                 memcpy(A_local->AS_local, A_global->AS, A_local->local_nnz * sizeof(float));
            }
        } else {
            A_local->IRP_local = NULL;
            A_local->JA_local = NULL;
            A_local->AS_local = NULL;
        }


        free(rows_per_rank);
        free(displs_rows);
        free(displs_nnz);

    } else { // Processi non-root
        MPI_Recv(&A_local->local_nrows, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&A_local->global_ncols, 1, MPI_INT, 0, 1, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&A_local->local_nnz, 1, MPI_INT, 0, 2, comm, MPI_STATUS_IGNORE);

        if (A_local->local_nrows > 0) {
            A_local->IRP_local = (int *)malloc((A_local->local_nrows + 1) * sizeof(int));
            if (A_local->local_nnz > 0) {
                A_local->JA_local = (int *)malloc(A_local->local_nnz * sizeof(int));
                A_local->AS_local = (float *)malloc(A_local->local_nnz * sizeof(float));
                MPI_Recv(A_local->JA_local, A_local->local_nnz, MPI_INT, 0, 3, comm, MPI_STATUS_IGNORE);
                MPI_Recv(A_local->AS_local, A_local->local_nnz, MPI_FLOAT, 0, 4, comm, MPI_STATUS_IGNORE);
            } else { // Righe vuote, ma local_nrows > 0
                A_local->JA_local = NULL;
                A_local->AS_local = NULL;
            }
            MPI_Recv(A_local->IRP_local, A_local->local_nrows + 1, MPI_INT, 0, 5, comm, MPI_STATUS_IGNORE);
        } else { // Questo rank non ha righe
            A_local->IRP_local = NULL;
            A_local->JA_local = NULL;
            A_local->AS_local = NULL;
        }
    }
}


void mpi_spmv_csr(const CSRMatrix_local *A_local, const float *x_broadcasted, float *y_local_buff, float *y_global_on_root, int rank, int size, MPI_Comm comm) {
    // Ogni processo calcola la sua porzione di y
    // y_local_buff deve essere allocato con A_local->local_nrows elementi
    // x_broadcasted deve essere allocato con A_local->global_ncols elementi e già popolato

    if (A_local->local_nrows > 0) {
        // Usiamo una CSRMatrix temporanea per chiamare serial_spmv
        CSRMatrix temp_csr_for_serial;
        temp_csr_for_serial.nrows = A_local->local_nrows;
        temp_csr_for_serial.ncols = A_local->global_ncols;
        temp_csr_for_serial.nnz = A_local->local_nnz;
        temp_csr_for_serial.IRP = A_local->IRP_local;
        temp_csr_for_serial.JA = A_local->JA_local;
        temp_csr_for_serial.AS = A_local->AS_local;

        serial_spmv(&temp_csr_for_serial, x_broadcasted, y_local_buff);
    }

    // Raccogli i risultati su rank 0
    // Prima, rank 0 deve sapere quante righe riceverà da ogni processo
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
    }

    // Tutti i processi inviano il loro numero di righe locali a rank 0
    // Oppure, rank 0 può ricalcolarlo se ha salvato rows_per_rank da distribute_matrix
    // Per semplicità, usiamo MPI_Gather per raccogliere local_nrows.
    // Alternativa più efficiente: rank 0 calcola recvcounts e displs come ha fatto per la distribuzione.
    // Qui assumiamo che rank 0 ricalcoli/abbia già questa info.

    // Per MPI_Gatherv, rank 0 ha bisogno di sapere quanti elementi aspettarsi da ciascuno
    // Questa informazione è implicita in come sono state distribuite le righe
    if (rank == 0) {
        int M_global_total_rows = 0; // Per y_global_on_root
        int base_rows = 0;
        int remainder_rows = 0;

        // Dobbiamo sapere il numero totale di righe della matrice originale
        // Potremmo passarlo o dedurlo. Se y_global_on_root è già allocato,
        // allora A_global->nrows è noto. Se non c'è A_global qui, la cosa si complica.
        // Assumiamo che A_local->global_ncols sia stato propagato, ma M_global no.
        // La soluzione migliore è che rank 0 determini M_global dalla sua CSRMatrix A_global
        // e poi calcoli recvcounts e displs per MPI_Gatherv.

        // Se A_local non è stato popolato perché matrice vuota, questo blocco non deve fare nulla
        if (A_local->global_ncols > 0 || A_local->local_nrows > 0 ) { // Check per matrice non vuota
            // Per ora, un trucco: ogni processo invia la sua `local_nrows` a rank 0
            // Questo è inefficiente ma illustrativo. Una soluzione migliore è che rank 0 lo sappia già.
            int M_loc_temp = A_local->local_nrows;
            MPI_Gather(&M_loc_temp, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);

            displs[0] = 0;
            M_global_total_rows = recvcounts[0];
            for (int i = 1; i < size; i++) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
                M_global_total_rows += recvcounts[i];
            }
            // Ora y_global_on_root dovrebbe essere allocato con M_global_total_rows elementi su rank 0
            // se M_global_total_rows > 0
        } else if (y_global_on_root != NULL) { // Matrice vuota, ma y_global potrebbe essere stato allocato
            for(int i=0; i<size; ++i) recvcounts[i] = 0;
            for(int i=0; i<size; ++i) displs[i] = 0;
        }
    } else { // Non-root ranks
        int M_loc_temp = A_local->local_nrows;
        MPI_Gather(&M_loc_temp, 1, MPI_INT, NULL, 0, MPI_INT, 0, comm); // Invia a rank 0
    }


    // Ora esegui il Gatherv effettivo di y_local_buff
    // Assicurati che y_global_on_root sia allocato su rank 0 con la dimensione corretta
    // e che y_local_buff sia allocato su tutti i rank con A_local->local_nrows
    if (A_local->local_nrows > 0 || (rank == 0 && y_global_on_root != NULL)) { // Se c'è qualcosa da inviare/ricevere
         MPI_Gatherv(y_local_buff, A_local->local_nrows, MPI_FLOAT,
                    y_global_on_root, recvcounts, displs, MPI_FLOAT,
                    0, comm);
    }


    if (rank == 0) {
        if (recvcounts) free(recvcounts);
        if (displs) free(displs);
    }
}