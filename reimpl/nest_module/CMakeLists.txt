cmake_minimum_required(VERSION 3.19)

if(NOT with-nest)
  message(FATAL_ERROR "-Dwith-nest=/path/to/nest-config is required")
endif()

execute_process(
  COMMAND ${with-nest} --compiler
  RESULT_VARIABLE nest_config_status
  OUTPUT_VARIABLE nest_compiler
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT nest_config_status EQUAL 0)
  message(FATAL_ERROR "Could not execute ${with-nest}")
endif()
set(CMAKE_CXX_COMPILER "${nest_compiler}")
project(zd3module CXX)

execute_process(
  COMMAND ${with-nest} --cflags
  OUTPUT_VARIABLE nest_cxx_flags
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${with-nest} --includes
  OUTPUT_VARIABLE nest_include_flags
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${with-nest} --libs
  OUTPUT_VARIABLE nest_link_flags
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

separate_arguments(nest_include_flags NATIVE_COMMAND "${nest_include_flags}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(zd3module MODULE
  zd3_module.cpp
  zd3_midpoint_neuron.cpp
  zd3_triplet_synapse.cpp
)
target_compile_definitions(zd3module PRIVATE LTX_MODULE)
target_compile_options(zd3module PRIVATE ${nest_include_flags})
set_target_properties(zd3module PROPERTIES
  COMPILE_FLAGS "${nest_cxx_flags}"
  LINK_FLAGS "${nest_link_flags}"
  PREFIX ""
  OUTPUT_NAME "zd3module"
)
