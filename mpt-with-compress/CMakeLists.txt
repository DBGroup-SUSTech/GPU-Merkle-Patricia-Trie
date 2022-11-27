cmake_minimum_required(VERSION 3.20)
project(GpuMPT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


find_package(fmt)
find_package(GTest)

include(GoogleTest)

set(SRC_PATH ${PROJECT_SOURCE_DIR}/src)
set(INC_PATH ${PROJECT_SOURCE_DIR}/include)
set(TEST_PATH ${PROJECT_SOURCE_DIR}/test)

file(GLOB_RECURSE SRC_FILES "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE TEST_FILES "test/*.cu" "test/*.cpp")

foreach(test_file IN LISTS TEST_FILES)
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file} ${SRC_FILES})
        target_link_libraries(${test_name} GTest::gtest_main)
        gtest_discover_tests(${test_name})
endforeach()

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

include_directories(
        PUBLIC ${INC_PATH}
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        PUBLIC ${GTEST_INCLUDE_DIRS})
link_libraries(
        PUBLIC cuda 
        PUBLIC fmt::fmt)