# Install script for directory: /home/ewan/Desktop/visualizer/benchmarks/polybench/OpenCL

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "DEBUG")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/2DCONV/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/2MM/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/3DCONV/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/3MM/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/ATAX/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/ATAX2/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/ATAX3/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/BICG/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/CORR/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/CORR2/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/COVAR/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/FDTD-2D/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/GEMM/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/GESUMMV/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/GRAMSCHM/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/MVT/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K/cmake_install.cmake")
  INCLUDE("/home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYRK/cmake_install.cmake")

ENDIF(NOT CMAKE_INSTALL_LOCAL_ONLY)

