cmake_minimum_required(VERSION 3.20)
project(GPHO VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find LLVM Package Configuration (LLVM 18)
find_package(LLVM REQUIRED CONFIG)
if(NOT LLVM_VERSION_MAJOR EQUAL 18)
    message(WARNING "GPHO targets LLVM 18. Found LLVM version ${LLVM_PACKAGE_VERSION}")
endif()
message(STATUS "GPHO: Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "GPHO: Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "GPHO: LLVM Include Dirs: ${LLVM_INCLUDE_DIRS}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pass)

# Build GPHOPass loadable module
add_library(GPHOPass MODULE
    pass/GPHOAnalysisPass.h
    pass/GPHOAnalysisPass.cpp
)

target_compile_options(GPHOPass PRIVATE
    -fno-rtti
    -fno-exceptions
    -Wall
    -Wextra
)

set_target_properties(GPHOPass PROPERTIES
    PREFIX ""
    OUTPUT_NAME "GPHOPass"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pass"
)

if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(GPHOPass PRIVATE -fno-rtti)
endif()

message(STATUS "GPHO: Configured build for GPHOPass LLVM Plugin.")
