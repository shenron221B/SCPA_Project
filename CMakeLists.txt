cmake_minimum_required(VERSION 3.18)
project(SCPA_Project LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Imposta architettura CUDA
set(CMAKE_CUDA_ARCHITECTURES 75)

# OpenMP
find_package(OpenMP REQUIRED)

# Aggiungi eseguibile
add_executable(SCPA
        main.cpp
        src/mm_reader.c
        src/csr_utils.c
        src/serial.c
        src/openmp_spmv.c
        CUDA/src/cuda_spmv.cu
)

# Include directory
target_include_directories(SCPA PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/include
)

# Linka librerie
target_link_libraries(SCPA
        OpenMP::OpenMP_C
        OpenMP::OpenMP_CXX
        cudart
)
