cmake_minimum_required(VERSION 3.10) # Mantenuto 3.10 per compatibilità
project(SCPA_Project C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)

# Aggiungi l'eseguibile
add_executable(SCPA
        main.c
        src/mm_reader.c
        src/csr_utils.c
        src/serial.c
        src/openmp_spmv.c
)

# Imposta le directory di include
target_include_directories(SCPA PUBLIC
        ${OpenMP_C_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Linka le librerie
target_link_libraries(SCPA PUBLIC
        OpenMP::OpenMP_C
)