set(EXE_NAME scheduler)

set(BUILD_DIR ${CMAKE_BINARY_DIR}/${TOOLS_DIR}/${SCHEDULER_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
# Src files.

file(GLOB SOURCE_FILES_LIST "${SCHEDULER_PATH}/*.cpp")
add_executable(${EXE_NAME} ${SOURCE_FILES_LIST})
