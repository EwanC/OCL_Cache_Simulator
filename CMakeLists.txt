cmake_minimum_required (VERSION 2.8)
set(CMAKE_CXX_COMPILER "g++")

set(PROJECT_NAME visualizer)
project(PROJECT_NAME)

# Add the appropriate include directory here.
set(OPENCL_INCLUDE_PATH "$ENV{AMDAPPSDKROOT}/include/CL")

# Add the appropriate lib directory here.
set(OPENCL_LIB_DIR_PATH "$ENV{AMDAPPSDKROOT}/lib/x86")

set(BENCHMARKS_DIR "benchmarks")
set(TOOLS_DIR "tools")
set(SUPPORT_DIR "support")

set(BENCHMARKS_PATH ${PROJECT_SOURCE_DIR}/${BENCHMARKS_DIR})
set(TOOLS_PATH ${PROJECT_SOURCE_DIR}/${TOOLS_DIR})
set(SUPPORT_PATH ${PROJECT_SOURCE_DIR}/${SUPPORT_DIR})

# Variables needed by benchmarks.
set(BENCH_SUPPORT_LIB "bench_support")
set(BENCH_SUPPORT_INCLUDE_DIR "include")
set(BENCH_SUPPORT_LIB_DIR "bench_support")

set(OPENCL_WRAPPER_LIB "OpenCLWrapper")
set(OPENCL_WRAPPER_DIR "wrapper")
set(OPENCL_WRAPPER_LIB_DIR "lib")
set(OPENCL_WRAPPER_CODE_LIB_DIR "OpenCLWrapper")
set(OPENCL_WRAPPER_UTILS_LIB "OpenCLWrapperUtils")
set(OPENCL_WRAPPER_UTILS_LIB_DIR "Utils")
set(OPENCL_WRAPPER_INCLUDE_DIR "include")

set(SCRIPT_DIR "scripts")
set(SCRIPT_PATH ${SUPPORT_PATH}/${SCRIPT_DIR})


#-------------------------------------------------
# Check for executables
#

find_file(CLANG_EXE "clang" PATHS ENV PATH)
if(NOT EXISTS ${CLANG_EXE})
  message(FATAL_ERROR " Please install clang compiler front end, and add to $PATH")
endif(NOT EXISTS ${CLANG_EXE})

find_file(OPT_EXE "opt" PATHS ENV PATH)
if(NOT EXISTS ${OPT_EXE})
  message(FATAL_ERROR " Please install llvm optimizer, opt, and add to $PATH")
endif(NOT EXISTS ${OPT_EXE})


find_file(RSCRIPT_EXE "Rscript" PATHS ENV PATH)
if(NOT EXISTS ${RSCRIPT_EXE})
  message(FATAL_ERROR " Please install R graphing tool(www.r-project.org/), and add to $PATH")
endif(NOT EXISTS ${RSCRIPT_EXE})

find_file(AXTOR_EXE "axtor" PATHS ENV PATH)
if(NOT EXISTS ${AXTOR_EXE})
  message(FATAL_ERROR " Please install axtor compiler back end(bitbucket.org/gnarf/axtor/), and add to $PATH")
endif(NOT EXISTS ${AXTOR_EXE})


# -------------------------------------------------------------------------------
# Build configuration.
# Set a default build type for single-configuration.
# To set the build type from the command line use the -DCMAKE_BUILD_TYPE
# option.

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE DEBUG)
endif(CMAKE_BUILD_TYPE STREQUAL "")

# Set compilation options.
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # Add specific DEBUG options.
  add_definitions("-DDEBUG")
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

# ------------------------------------------------------------------------------
# Library discovery.
find_library(OPENCL_LIB OpenCL ${OPENCL_LIB_DIR_PATH} DOC "The OpenCL Library")
if(NOT EXISTS ${OPENCL_LIB})
  message(FATAL_ERROR "OpenCL library not found")
endif(NOT EXISTS ${OPENCL_LIB})

add_subdirectory(${TOOLS_PATH})
add_subdirectory(${BENCHMARKS_PATH})
add_subdirectory(${SUPPORT_PATH})
