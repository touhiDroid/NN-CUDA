cmake_minimum_required(VERSION 3.20)

project(DNN)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
# add_executable(DNN main.cpp)
add_executable(DNN main.cpp cuda.cpp)

#set(CMAKE_CUDA_STANDARD 14)
#project(DNN_CUDA CUDA)
#add_executable(DNN_CUDA main.cu)
# set_target_properties(DNN_CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)