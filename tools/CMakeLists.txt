set(SCHEDULER_DIR "scheduler")
set(CACHESIM_DIR "cacheSimulator")

set(SCHEDULER_PATH ${TOOLS_PATH}/${SCHEDULER_DIR})
set(CACHESIM_PATH ${TOOLS_PATH}/${CACHESIM_DIR})

add_subdirectory(${SCHEDULER_PATH})
add_subdirectory(${CACHESIM_PATH})
