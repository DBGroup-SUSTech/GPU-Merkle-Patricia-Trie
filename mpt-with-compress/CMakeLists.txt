cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(GpuMPT LANGUAGES CXX CUDA)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Enable ASAN only as you need. Please notice
#       1. This project do not pass leak-santizer
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

find_package(GTest)
find_package(LibXml2 REQUIRED)
include(GoogleTest)
find_library(CRYPTOPP_LIB cryptopp)

add_compile_definitions(PROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\")

set(SRC_PATH ${PROJECT_SOURCE_DIR}/src)
set(INC_PATH ${PROJECT_SOURCE_DIR}/include)
set(TEST_PATH ${PROJECT_SOURCE_DIR}/test)
message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

file(GLOB_RECURSE SRC_FILES "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE TEST_FILES "test/*.cu" "test/*.cpp")

foreach(test_file IN LISTS TEST_FILES)
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file} ${SRC_FILES})
        target_include_directories(${test_name}
                PUBLIC ${INC_PATH}
                PUBLIC ${LIBXML2_INCLUDE_DIR}
                PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})  
        target_link_libraries(${test_name} 
                PRIVATE cuda 
                PRIVATE GTest::gtest_main
                PRIVATE ${CRYPTOPP_LIB}
                PRIVATE ${LIBXML2_LIBRARIES}
        ) 

        gtest_discover_tests(${test_name})
endforeach()

# Static Library
add_library(gmpt STATIC ${PROJECT_SOURCE_DIR}/lib/libgmpt.cu ${SRC_FILES})
target_include_directories(gmpt
        PUBLIC ${INC_PATH}
        PUBLIC ${LIBXML2_INCLUDE_DIR}
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})  
target_link_libraries(gmpt
        PUBLIC STATIC cuda 
        PUBLIC STATIC GTest::gtest_main
        PUBLIC STATIC ${CRYPTOPP_LIB}
        PUBLIC STATIC ${LIBXML2_LIBRARIES}) 
