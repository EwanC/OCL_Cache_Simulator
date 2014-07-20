set(SRC_DIR "src")
set(BENCH_SUPPORT_INCLUDE_DIR "include") 

# Full paths.
set(SRC_PATH ${BENCH_LIB_PATH}/${SRC_DIR}/)
set(INCLUDE_PATH ${BENCH_LIB_PATH}/${BENCH_SUPPORT_INCLUDE_DIR}/)

# Src files.
file(GLOB SOURCE_FILES_LIST "${SRC_PATH}/*.cpp")

add_library(${BENCH_SUPPORT_LIB} SHARED ${SOURCE_FILES_LIST})

# Include directories.
include_directories("${INCLUDE_PATH}")
