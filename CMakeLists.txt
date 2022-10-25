cmake_minimum_required(VERSION 3.20)
project(GpuMPT CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)

find_package(fmt)

set(SRC_PATH ${PROJECT_SOURCE_DIR}/src)
set(INC_PATH ${PROJECT_SOURCE_DIR}/include)

add_executable(PerfTest ${SRC_PATH}/PerfTest.cpp ${INC_PATH})

include_directories(
        PUBLIC ${INC_PATH}
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
link_libraries(
        PUBLIC cuda 
        PUBLIC fmt::fmt)

set_target_properties(PerfTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)