cmake_minimum_required(VERSION 3.20)
project(DNN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

add_executable(DNN main.cpp)

project(DNN_CUDA)
#set(CUDA_TOOLKIT_ROOT_DIR "/Users/touhiduzzamm/cuda-gdb-darwin-11.5/bin")
#find_package(CUDA REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)
#file(GLOB  cu  *.cu)
#cuda_add_executable(DNN_CUDA main.cu ${cu})
add_executable(DNN_CUDA cuda.cpp)