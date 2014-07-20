set(MM_DIR "mm")
set(MT_DIR "mt")
set(MV_DIR "mv")
  
set(MM_PATH ${MATRIX_PATH}/${MM_DIR}) 
set(MT_PATH ${MATRIX_PATH}/${MT_DIR})
set(MV_PATH ${MATRIX_PATH}/${MV_DIR})
  
add_subdirectory(${MM_PATH})
add_subdirectory(${MT_PATH})
add_subdirectory(${MV_PATH})

