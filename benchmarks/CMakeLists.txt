set(POLYBENCH_DIR "polybench")
set(MATRIX_DIR "matrix")
set(PARBOIL_DIR "parboil")

set(POLYBENCH_PATH ${BENCHMARKS_PATH}/${POLYBENCH_DIR})
set(MATRIX_PATH ${BENCHMARKS_PATH}/${MATRIX_DIR})
set(PARBOIL_PATH ${BENCHMARKS_PATH}/${PARBOIL_DIR})

add_subdirectory(${MATRIX_PATH})
add_subdirectory(${POLYBENCH_PATH})
add_subdirectory(${PARBOIL_PATH})
