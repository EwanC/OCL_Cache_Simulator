# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ewan/Desktop/visualizer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ewan/Desktop/visualizer/build

# Include any dependencies generated for this target.
include benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/depend.make

# Include the progress variables for this target.
include benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/progress.make

# Include the compile flags for this target's objects.
include benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/flags.make

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/flags.make
benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o: ../benchmarks/polybench/OpenCL/SYR2K/syr2k.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ewan/Desktop/visualizer/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o"
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/SYR2K.dir/syr2k.cpp.o -c /home/ewan/Desktop/visualizer/benchmarks/polybench/OpenCL/SYR2K/syr2k.cpp

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SYR2K.dir/syr2k.cpp.i"
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ewan/Desktop/visualizer/benchmarks/polybench/OpenCL/SYR2K/syr2k.cpp > CMakeFiles/SYR2K.dir/syr2k.cpp.i

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SYR2K.dir/syr2k.cpp.s"
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ewan/Desktop/visualizer/benchmarks/polybench/OpenCL/SYR2K/syr2k.cpp -o CMakeFiles/SYR2K.dir/syr2k.cpp.s

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o.requires:
.PHONY : benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o.requires

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o.provides: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o.requires
	$(MAKE) -f benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/build.make benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o.provides.build
.PHONY : benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o.provides

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o.provides.build: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/flags.make
benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o: ../benchmarks/polybench/common/src/Utils.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ewan/Desktop/visualizer/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o"
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o -c /home/ewan/Desktop/visualizer/benchmarks/polybench/common/src/Utils.cpp

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.i"
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ewan/Desktop/visualizer/benchmarks/polybench/common/src/Utils.cpp > CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.i

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.s"
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ewan/Desktop/visualizer/benchmarks/polybench/common/src/Utils.cpp -o CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.s

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o.requires:
.PHONY : benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o.requires

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o.provides: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o.requires
	$(MAKE) -f benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/build.make benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o.provides.build
.PHONY : benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o.provides

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o.provides.build: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/flags.make
benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o: ../benchmarks/polybench/common/src/MathUtils.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ewan/Desktop/visualizer/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o"
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o -c /home/ewan/Desktop/visualizer/benchmarks/polybench/common/src/MathUtils.cpp

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.i"
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ewan/Desktop/visualizer/benchmarks/polybench/common/src/MathUtils.cpp > CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.i

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.s"
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ewan/Desktop/visualizer/benchmarks/polybench/common/src/MathUtils.cpp -o CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.s

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o.requires:
.PHONY : benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o.requires

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o.provides: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o.requires
	$(MAKE) -f benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/build.make benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o.provides.build
.PHONY : benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o.provides

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o.provides.build: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o

# Object files for target SYR2K
SYR2K_OBJECTS = \
"CMakeFiles/SYR2K.dir/syr2k.cpp.o" \
"CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o" \
"CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o"

# External object files for target SYR2K
SYR2K_EXTERNAL_OBJECTS =

benchmarks/polybench/OpenCL/SYR2K/SYR2K: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o
benchmarks/polybench/OpenCL/SYR2K/SYR2K: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o
benchmarks/polybench/OpenCL/SYR2K/SYR2K: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o
benchmarks/polybench/OpenCL/SYR2K/SYR2K: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/build.make
benchmarks/polybench/OpenCL/SYR2K/SYR2K: support/wrapper/lib/Utils/libOpenCLWrapperUtils.so
benchmarks/polybench/OpenCL/SYR2K/SYR2K: support/wrapper/lib/OpenCLWrapper/libOpenCLWrapper.so
benchmarks/polybench/OpenCL/SYR2K/SYR2K: support/bench_support/libbench_support.so
benchmarks/polybench/OpenCL/SYR2K/SYR2K: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
benchmarks/polybench/OpenCL/SYR2K/SYR2K: /usr/lib/x86_64-linux-gnu/libboost_system.so
benchmarks/polybench/OpenCL/SYR2K/SYR2K: support/wrapper/lib/Utils/libOpenCLWrapperUtils.so
benchmarks/polybench/OpenCL/SYR2K/SYR2K: /usr/lib/libOpenCL.so
benchmarks/polybench/OpenCL/SYR2K/SYR2K: /usr/lib/x86_64-linux-gnu/libboost_system.so
benchmarks/polybench/OpenCL/SYR2K/SYR2K: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable SYR2K"
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SYR2K.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/build: benchmarks/polybench/OpenCL/SYR2K/SYR2K
.PHONY : benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/build

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/requires: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/syr2k.cpp.o.requires
benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/requires: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/Utils.cpp.o.requires
benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/requires: benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/__/__/common/src/MathUtils.cpp.o.requires
.PHONY : benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/requires

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/clean:
	cd /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K && $(CMAKE_COMMAND) -P CMakeFiles/SYR2K.dir/cmake_clean.cmake
.PHONY : benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/clean

benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/depend:
	cd /home/ewan/Desktop/visualizer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ewan/Desktop/visualizer /home/ewan/Desktop/visualizer/benchmarks/polybench/OpenCL/SYR2K /home/ewan/Desktop/visualizer/build /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K /home/ewan/Desktop/visualizer/build/benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : benchmarks/polybench/OpenCL/SYR2K/CMakeFiles/SYR2K.dir/depend

