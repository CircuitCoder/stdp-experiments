cmake_minimum_required(VERSION 3.20)
project(stdp_trace_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(trace-sim src/main.cpp)
target_include_directories(trace-sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/json/single_include)
target_compile_options(trace-sim PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME trace-sim-self-test COMMAND trace-sim --self-test)
