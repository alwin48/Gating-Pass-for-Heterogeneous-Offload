cmake_minimum_required(VERSION 3.20)
project(GPHOPass LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM 18 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

add_library(GPHOPass MODULE GPHOAnalysisPass.cpp)
target_compile_options(GPHOPass PRIVATE -fno-rtti -fno-exceptions)
set_target_properties(GPHOPass PROPERTIES PREFIX "")

# Avoid linking libLLVM.so when building a loadable module on Ubuntu packages.
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(GPHOPass PRIVATE -fno-rtti)
endif()
