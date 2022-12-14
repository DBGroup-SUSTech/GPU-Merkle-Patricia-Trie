cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(GpuMPT LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Enable ASAN only as you need. Please notice
#       1. This project do not pass leak-santizer
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

find_package(fmt)
find_package(GTest)

include(GoogleTest)

set(CRYPTOPP_INCLUDE_DIR "/usr/local/include/cryptopp/*.h")
find_path(CRYPTOPP_INCLUDE_DIR cryptopp/*.h PATHS )
include_directories(${CRYPTOPP_INCLUDE_DIR})

find_library(CRYPTOPP_LIB cryptopp)
message(STATUS "finding ${CRYPTOPP_LIB}")

set(SRC_PATH ${PROJECT_SOURCE_DIR}/src)
set(INC_PATH ${PROJECT_SOURCE_DIR}/include)
set(TEST_PATH ${PROJECT_SOURCE_DIR}/test)

file(GLOB_RECURSE SRC_FILES "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE TEST_FILES "test/*.cu" "test/*.cpp")

foreach(test_file IN LISTS TEST_FILES)
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file} ${SRC_FILES})
        target_include_directories(${test_name}
                PUBLIC ${INC_PATH})  
        target_link_libraries(${test_name} 
                PRIVATE cuda 
                PRIVATE fmt::fmt 
                PRIVATE GTest::gtest_main
                ${CRYPTOPP_LIB}
        ) 

        gtest_discover_tests(${test_name})
endforeach()
