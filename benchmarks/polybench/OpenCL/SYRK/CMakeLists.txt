get_filename_component(CURRENT_BENCH ${CMAKE_CURRENT_SOURCE_DIR} NAME) 
set(EXE_NAME ${CURRENT_BENCH})

set(BUILD_DIR ${CMAKE_BINARY_DIR}/${BENCHMARKS_DIR}/${POLYBENCH_DIR}/${OPENCL_DIR}/${CURRENT_BENCH})
set(OPENCL_WRAPPER_INCLUDE_PATH ${SUPPORT_PATH}/${OPENCL_WRAPPER_DIR}/${OPENCL_WRAPPER_INCLUDE_DIR}/${OPENCL_WRAPPER_CODE_LIB_DIR})
set(BENCH_SUPPORT_INCLUDE_PATH ${SUPPORT_PATH}/${BENCH_SUPPORT_LIB_DIR}/${BENCH_SUPPORT_INCLUDE_DIR})

file(GLOB SRC_FILES_LIST "${SYRK_PATH}/*.cpp")
file(GLOB COMMON_FILES_LIST ${COMMON_SRC_PATH}/*.cpp)

configure_file("${COMMON_INCLUDE_PATH}/SystemConfig.h.cmake"
               "${CMAKE_BINARY_DIR}/${BENCHMARKS_DIR}/${POLYBENCH_DIR}/common/include/SystemConfig.h")

set(KERNEL_DIRECTORY "${SYRK_PATH}/")
configure_file("${SYRK_PATH}/SystemConfiguration.h.cmake"
               "${BUILD_DIR}/SystemConfiguration.h")


include_directories("${CMAKE_BINARY_DIR}/${BENCHMARKS_DIR}/${POLYBENCH_DIR}/common/include/")
include_directories(${COMMON_INCLUDE_PATH})
include_directories(${OPENCL_WRAPPER_INCLUDE_PATH})
include_directories(${BENCH_SUPPORT_INCLUDE_PATH})
include_directories(${BUILD_DIR})

add_executable(${EXE_NAME} ${SRC_FILES_LIST} ${COMMON_FILES_LIST})
set_target_properties(${EXE_NAME} PROPERTIES COMPILE_FLAGS -DKERNEL_PATH="\\"${CMAKE_CURRENT_SOURCE_DIR}\\"")

#-------------------------------------------------------------------------------
# Libraries.

# OpenCL library.
target_link_libraries(${EXE_NAME} ${OPENCL_LIB_PATH})

# OpenCLWrapper Utils library.
link_directories(${CMAKE_BINARY_DIR}/${SUPPORT_DIR}/${OPENCL_WRAPPER_DIR}/${OPENCL_WRAPPER_LIB_DIR}/${OPENCL_WRAPPER_UTILS_LIB_DIR})
target_link_libraries(${EXE_NAME} ${OPENCL_WRAPPER_UTILS_LIB})
                    
# OpenCLWrapper library.              
link_directories(${CMAKE_BINARY_DIR}/${SUPPORT_DIR}/${OPENCL_WRAPPER_DIR}/${OPENCL_WRAPPER_LIB_DIR}/${OPENCL_WRAPPER_CODE_LIB_DIR})
target_link_libraries(${EXE_NAME} ${OPENCL_WRAPPER_LIB})

# BenchSupport library.
link_directories(${CMAKE_BINARY_DIR}/${SUPPORT_DIR}/${BENCH_SUPPORT_LIB_DIR})
target_link_libraries(${EXE_NAME} ${BENCH_SUPPORT_LIB})

# Boost FileSystem library.
find_library(BOOST_FILESYSTEM_LIBRARY_PATH boost_filesystem ${BOOST_LIB_PATHS}
             DOC "The Boost FileSystem Library")
if(NOT EXISTS ${BOOST_FILESYSTEM_LIBRARY_PATH})
  message(FATAL_ERROR "Boost-FileSystem library not found")
endif(NOT EXISTS ${BOOST_FILESYSTEM_LIBRARY_PATH})
target_link_libraries(${EXE_NAME} ${BOOST_FILESYSTEM_LIBRARY_PATH})

# Boost System library.
find_library(BOOST_SYSTEM_LIBRARY_PATH boost_system ${BOOST_LIB_PATHS}
              DOC "The Boost-System Library")
if(NOT EXISTS ${BOOST_SYSTEM_LIBRARY_PATH})
   message(FATAL_ERROR "Boost-System library not found")
endif(NOT EXISTS ${BOOST_SYSTEM_LIBRARY_PATH})
target_link_libraries(${EXE_NAME} ${BOOST_SYSTEM_LIBRARY_PATH})
