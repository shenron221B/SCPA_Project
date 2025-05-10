#include "../include/openmp_spmv.h"
#include <omp.h>    // Necessario per le direttive OpenMP e le funzioni runtime
#include <stdio.h>  // Per eventuali printf di debug

/**
 * @brief Calcola il prodotto matrice-vettore y = Ax per una matrice sparsa in formato CSR
 *        utilizzando OpenMP per la parallelizzazione del ciclo sulle righe.
 *
 * Dettagli dell'implementazione:
 * - La parallelizzazione avviene sul ciclo esterno che itera sulle righe della matrice A.
 * - Ogni thread OpenMP gestisce un sottoinsieme di righe.
 * - La variabile 'sum' per l'accumulo parziale è dichiarata privata per ogni thread
 *   all'interno del ciclo parallelizzato per evitare race condition.
 * - Si utilizza una schedule 'static' per la distribuzione delle iterazioni (righe) ai thread.
 *   Una schedule 'dynamic' o 'guided' potrebbe essere considerata per matrici con
 *   distribuzione dei non-zeri per riga molto irregolare, ma 'static' è spesso un buon punto di partenza.
 *
 * @param A Puntatore alla matrice CSR (CSRMatrix). La matrice non viene modificata.
 * @param x Puntatore al vettore di input x. Non viene modificato.
 * @param y Puntatore al vettore di output y. Verrà riempito con il risultato.
 * @param num_threads Numero di thread OpenMP da utilizzare per il calcolo.
 *                    Se num_threads <= 0, OpenMP userà il numero di thread di default.
 */
void openmp_spmv_csr(const CSRMatrix *A, const float *x, float *y, int num_threads) {
    // Verifica che i puntatori non siano NULL per evitare accessi invalidi
    if (!A || !A->IRP || !A->JA || !A->AS || !x || !y) {
        fprintf(stderr, "Errore [openmp_spmv_csr]: uno o più argomenti sono NULL.\n");
        return;
    }
    // Verifica la coerenza delle dimensioni (semplificata, si assume che x e y siano allocati correttamente)
    // if (A->ncols <= 0 || A->nrows <= 0) {
    //     fprintf(stderr, "Errore [openmp_spmv_csr]: dimensioni della matrice non valide.\n");
    //     return;
    // }


    // Imposta il numero di thread da usare, se specificato e valido
    // Se num_threads è <= 0, OpenMP usa il numero di default (spesso il numero di core disponibili)
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    // Direttiva OpenMP per parallelizzare il ciclo for.
    // - 'parallel for': indica che il ciclo seguente deve essere eseguito in parallelo dai thread.
    // - 'shared(A, x, y)': specifica che le variabili A, x, y sono condivise tra tutti i thread.
    //                      A e x sono letti, y è scritto (ogni thread scrive nella sua porzione).
    // - 'private(i, j, sum)': specifica che le variabili i (contatore del ciclo esterno),
    //                         j (contatore del ciclo interno) e sum (accumulatore locale)
    //                         sono private per ogni thread. Ogni thread avrà la sua copia.
    // - 'schedule(static)': le iterazioni (righe) vengono divise in blocchi di dimensione circa
    //                       (numero_righe / numero_thread) e assegnate staticamente ai thread.
    //                       Per matrici molto sbilanciate, 'dynamic' o 'guided' potrebbero essere migliori,
    //                       ma 'static' ha meno overhead.
    #pragma omp parallel for shared(A, x, y) schedule(static)
    for (int i = 0; i < A->nrows; i++) {
        float sum = 0.0f; // Variabile privata per ogni thread per l'accumulo
        // Il ciclo interno è eseguito serialmente da ciascun thread per le righe assegnategli.
        for (int k = A->IRP[i]; k < A->IRP[i + 1]; k++) {
            // A->JA[k] è l'indice di colonna dell'elemento non-zero
            // A->AS[k] è il valore dell'elemento non-zero
            // x[A->JA[k]] è l'elemento corrispondente del vettore x
            sum += A->AS[k] * x[A->JA[k]];
        }
        y[i] = sum; // Ogni thread scrive nel suo elemento y[i] corrispondente
    }
    // La barriera implicita alla fine del costrutto 'parallel for' assicura che
    // tutti i thread abbiano completato il loro lavoro prima di proseguire.
}