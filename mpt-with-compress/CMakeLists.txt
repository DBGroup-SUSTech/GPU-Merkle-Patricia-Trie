cmake_minimum_required(VERSION 3.20)
project(GpuMPT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(fmt)

set(SRC_PATH ${PROJECT_SOURCE_DIR}/src)
set(INC_PATH ${PROJECT_SOURCE_DIR}/include)
set(TEST_PATH ${PROJECT_SOURCE_DIR}/test)

file(GLOB SRC_FILES "src/*.cu" "src/*.cpp")
# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

# add_executable(PerfTest ${TEST_PATH}/perf_test.cu ${SRC_FILES})
# add_executable(HashTest ${TEST_PATH}/hash_test.cu ${SRC_FILES})
# add_executable(HashPerf ${TEST_PATH}/hash_perf.cu ${SRC_FILES})
# add_executable(HashSIMD ${TEST_PATH}/hash_SIMD.cu ${SRC_FILES})

# target_compile_options(HashSIMD PUBLIC -maxrregcount 64)

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

include_directories(
        PUBLIC ${INC_PATH}
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
link_libraries(
        PUBLIC cuda 
        PUBLIC fmt::fmt)

# set_target_properties(PerfTest PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON)
