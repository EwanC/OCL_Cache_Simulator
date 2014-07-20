set(SRC_DIR "src")
set(INCLUDE_DIR "include")

# Src files.
file(GLOB SRC_FILE_LIST "${CODE_LIB_PATH}/${SRC_DIR}/*.cpp")

# Include directories.
include_directories(${INCLUDE_PATH})
include_directories("${CODE_LIB_PATH}/${INCLUDE_DIR}")
include_directories(${OPENCL_INCLUDE_PATH})
include_directories(${BOOST_INCLUDE_PATHS})


# Library to build.
add_library(${OPENCL_WRAPPER_LIB} SHARED ${SRC_FILE_LIST})

target_link_libraries(${OPENCL_WRAPPER_LIB} ${OPENCL_LIB})

# Add dependency for OpenCLWrapperUtils library.
link_directories(${CMAKE_BINARY_DIR}/${SUPPORT_DIR}/${OPENCL_WRAPPER_DIR}/${OPENCL_WRAPPER_LIB_DIR}/${OPENCL_WRAPPER_UTILS_LIB_DIR})
target_link_libraries(${OPENCL_WRAPPER_LIB} ${OPENCL_WRAPPER_UTILS_LIB})


# Boost System library.
find_library(BOOST_SYSTEM_LIBRARY_PATH boost_system
             BOOST_LIB_PATHS
             DOC "The Boost-System Library")

if(NOT EXISTS ${BOOST_SYSTEM_LIBRARY_PATH})
   message(FATAL_ERROR "Boost-System library not found")
endif(NOT EXISTS ${BOOST_SYSTEM_LIBRARY_PATH})
target_link_libraries(${OPENCL_WRAPPER_LIB} ${BOOST_SYSTEM_LIBRARY_PATH})

