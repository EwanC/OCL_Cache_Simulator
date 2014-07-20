set(SRC_DIR "src")
set(INCLUDE_DIR "include")

# Src files.
file(GLOB SOURCE_FILE_LIST "${UTILS_PATH}/${SRC_DIR}/*.cpp")

# Include directories.
include_directories(${INCLUDE_PATH})

# Library to build.
add_library(${OPENCL_WRAPPER_UTILS_LIB} SHARED ${SOURCE_FILE_LIST})
