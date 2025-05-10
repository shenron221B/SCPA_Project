cmake_minimum_required(VERSION 3.10) # Mantenuto 3.10 per compatibilità
project(SCPA_Project C)

set(CMAKE_C_STANDARD 11)

# Trova il pacchetto MPI
find_package(MPI REQUIRED)

# Trova il pacchetto OpenMP (richiede CMake 3.9+)
find_package(OpenMP REQUIRED)

# Aggiungi l'eseguibile
add_executable(SCPA
        main.c
        src/mm_reader.c
        src/csr_utils.c
        src/serial.c
        src/mpi_spmv.c
        src/openmp_spmv.c   # NUOVO file sorgente
)

# Imposta le directory di include
target_include_directories(SCPA PUBLIC
        ${MPI_C_INCLUDE_PATH}
        ${OpenMP_C_INCLUDE_DIRS} # Include per OpenMP (di solito non serve, ma per completezza)
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Linka le librerie
target_link_libraries(SCPA PUBLIC
        ${MPI_C_LIBRARIES}
        OpenMP::OpenMP_C    # Modo moderno per linkare OpenMP
)