cmake_minimum_required(VERSION 3.18)
project(spmv_project LANGUAGES C CXX CUDA)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80" CACHE STRING "CUDA archs")
endif()
set(CUDAToolkit_ROOT "/opt/cuda/11.8")
set(CUDA_INCLUDE_DIR "/opt/cuda/11.8/include")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

include_directories(
        include
        CUDA/include
        ${CUDA_INCLUDE_DIR}
        ${CUDAToolkit_ROOT}/include
)

set(SOURCE_FILES
        src/csr_utils.c
        src/hll_utils.c
        src/mm_reader.c
        src/openmp_spmv.c
        src/openmp_hll_spmv.c
        src/serial.c
        main.cpp
)

# File CUDA
set(CUDA_FILES
        CUDA/src/cuda_spmv.cu
        CUDA/src/cuda_hll_spmv.cu
)

add_executable(spmv_exec
        ${SOURCE_FILES}
        ${CUDA_FILES}
)

set_target_properties(spmv_exec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    target_link_libraries(spmv_exec PUBLIC OpenMP::OpenMP_C)
endif()
