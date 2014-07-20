set(BFS_DIR "bfs")
set(STENCIL_DIR "stencil")
  
set(BFS_PATH ${PARBOIL_PATH}/${BFS_DIR}) 
set(STENCIL_PATH ${PARBOIL_PATH}/${STENCIL_DIR})
  
add_subdirectory(${BFS_PATH})
add_subdirectory(${STENCIL_PATH})
