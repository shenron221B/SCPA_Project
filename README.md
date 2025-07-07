# Prodotto Matrice Vettore Sparso (SpMV) Parallelo

Questo progetto, sviluppato per il corso di Sistemi di Calcolo Parallelo e Applicazioni, consiste nell'implementazione e nell'analisi prestazionale di un nucleo di calcolo per il prodotto matrice-vettore sparso ($y \leftarrow Ax$).

L'obiettivo è valutare e confrontare diverse strategie di parallelizzazione utilizzando i framework **OpenMP** (per CPU multi-core) e **CUDA** (per GPU NVIDIA). Sono stati implementati due formati di memorizzazione per matrici sparse, **Compressed Sparse Row (CSR)** e un formato ibrido basato su **ELLPACK (HLL)**, per analizzare l'impatto della struttura dati sulle performance.

## Compilazione ed Esecuzione

L'intero processo di compilazione ed esecuzione della suite di benchmark è automatizzato tramite lo script `build_and_run.sh`.

### 1. Sincronizzazione del Codice sul Server

Per prima cosa, occorre sincronizzare il codice locale con la directory di lavoro sul server. Sostituire `user@server` e `/path/to/project` con i propri dati.

```bash
    rsync -avz --delete --exclude='.git/' --exclude='build/' --exclude='*results*/' --exclude='plots/' --exclude='data/' ./ user@server:/path/to/project/
  ```
    
### 2. Connettersi e Lanciare lo Script

Connettersi al server via SSH. Lo script `build_and_run.sh` gestirà tutto il resto.

```bash
    # Connessione al server
    ssh user@server

    # Navigare alla directory del progetto
    cd /path/to/project/

    # Rendere lo script eseguibile 
    chmod +x build_and_run.sh

    # Lancio dello script di compilazione ed esecuzione
    ./build_and_run.sh
   ```

Lo script eseguirà tutti i test e salverà i file di log contenenti i dati di performance in una nuova directory con timestamp (es. `benchmark_results_YYYYMMDD_HHMMSS/`).

### 3. Analisi dei Risultati e Generazione Grafici

Questa fase viene eseguita in locale dopo aver scaricato i risultati.

1.  **Scaricare i Risultati:**
    Usare `rsync` per copiare la cartella dei risultati dal server in una directory locale `server_results/`. Assicurarsi di usare il nome corretto della cartella con il timestamp.

    ```bash
    rsync -avz user@server:/path/to/project/benchmark_results_YYYYMMDD_HHMMSS/ ./server_results/
    ```

2.  **Generare i Grafici:**
    Eseguire lo script Python che analizzerà tutti i file nella cartella `server_results/`, salverà i dati elaborati in formato `.csv` nella cartella `csv_results/` e genererà tutti i grafici delle performance nella cartella `plots/`.

    ```bash
    python analyze_results.py
    ```
    
### Esecuzione di un Test Singolo

Se necessario, è possibile eseguire una singola configurazione senza lanciare l'intera suite di benchmark. Per farlo occorre settare BENCHMARK a 0 nel main.

1.  **Compilare Manualmente sul Server:**
    ```bash
    # Dalla root del progetto sul server
    rm -rf build && mkdir build && cd build
    # Ddisabilitare la modalità benchmark in CMake
    cmake -DENABLE_BENCHMARK_SUITE=OFF ..
    make -j
    ```
    
2.  **Eseguire il Test:**
    Esempio per OpenMP CSR con 16 thread:
    ```bash
    # Dalla directory build
    ./spmv_exec csr_openmp 16
    ```