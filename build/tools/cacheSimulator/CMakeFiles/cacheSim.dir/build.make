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
include tools/cacheSimulator/CMakeFiles/cacheSim.dir/depend.make

# Include the progress variables for this target.
include tools/cacheSimulator/CMakeFiles/cacheSim.dir/progress.make

# Include the compile flags for this target's objects.
include tools/cacheSimulator/CMakeFiles/cacheSim.dir/flags.make

tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o: tools/cacheSimulator/CMakeFiles/cacheSim.dir/flags.make
tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o: ../tools/cacheSimulator/stats.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ewan/Desktop/visualizer/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cacheSim.dir/stats.cpp.o -c /home/ewan/Desktop/visualizer/tools/cacheSimulator/stats.cpp

tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cacheSim.dir/stats.cpp.i"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ewan/Desktop/visualizer/tools/cacheSimulator/stats.cpp > CMakeFiles/cacheSim.dir/stats.cpp.i

tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cacheSim.dir/stats.cpp.s"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ewan/Desktop/visualizer/tools/cacheSimulator/stats.cpp -o CMakeFiles/cacheSim.dir/stats.cpp.s

tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o.requires:
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o.requires

tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o.provides: tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o.requires
	$(MAKE) -f tools/cacheSimulator/CMakeFiles/cacheSim.dir/build.make tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o.provides.build
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o.provides

tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o.provides.build: tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o

tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o: tools/cacheSimulator/CMakeFiles/cacheSim.dir/flags.make
tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o: ../tools/cacheSimulator/parse.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ewan/Desktop/visualizer/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cacheSim.dir/parse.cpp.o -c /home/ewan/Desktop/visualizer/tools/cacheSimulator/parse.cpp

tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cacheSim.dir/parse.cpp.i"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ewan/Desktop/visualizer/tools/cacheSimulator/parse.cpp > CMakeFiles/cacheSim.dir/parse.cpp.i

tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cacheSim.dir/parse.cpp.s"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ewan/Desktop/visualizer/tools/cacheSimulator/parse.cpp -o CMakeFiles/cacheSim.dir/parse.cpp.s

tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o.requires:
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o.requires

tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o.provides: tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o.requires
	$(MAKE) -f tools/cacheSimulator/CMakeFiles/cacheSim.dir/build.make tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o.provides.build
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o.provides

tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o.provides.build: tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o

tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o: tools/cacheSimulator/CMakeFiles/cacheSim.dir/flags.make
tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o: ../tools/cacheSimulator/cache.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ewan/Desktop/visualizer/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cacheSim.dir/cache.cpp.o -c /home/ewan/Desktop/visualizer/tools/cacheSimulator/cache.cpp

tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cacheSim.dir/cache.cpp.i"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ewan/Desktop/visualizer/tools/cacheSimulator/cache.cpp > CMakeFiles/cacheSim.dir/cache.cpp.i

tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cacheSim.dir/cache.cpp.s"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ewan/Desktop/visualizer/tools/cacheSimulator/cache.cpp -o CMakeFiles/cacheSim.dir/cache.cpp.s

tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o.requires:
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o.requires

tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o.provides: tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o.requires
	$(MAKE) -f tools/cacheSimulator/CMakeFiles/cacheSim.dir/build.make tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o.provides.build
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o.provides

tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o.provides.build: tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o

tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o: tools/cacheSimulator/CMakeFiles/cacheSim.dir/flags.make
tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o: ../tools/cacheSimulator/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ewan/Desktop/visualizer/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/cacheSim.dir/main.cpp.o -c /home/ewan/Desktop/visualizer/tools/cacheSimulator/main.cpp

tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cacheSim.dir/main.cpp.i"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ewan/Desktop/visualizer/tools/cacheSimulator/main.cpp > CMakeFiles/cacheSim.dir/main.cpp.i

tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cacheSim.dir/main.cpp.s"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ewan/Desktop/visualizer/tools/cacheSimulator/main.cpp -o CMakeFiles/cacheSim.dir/main.cpp.s

tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o.requires:
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o.requires

tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o.provides: tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o.requires
	$(MAKE) -f tools/cacheSimulator/CMakeFiles/cacheSim.dir/build.make tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o.provides.build
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o.provides

tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o.provides.build: tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o

# Object files for target cacheSim
cacheSim_OBJECTS = \
"CMakeFiles/cacheSim.dir/stats.cpp.o" \
"CMakeFiles/cacheSim.dir/parse.cpp.o" \
"CMakeFiles/cacheSim.dir/cache.cpp.o" \
"CMakeFiles/cacheSim.dir/main.cpp.o"

# External object files for target cacheSim
cacheSim_EXTERNAL_OBJECTS =

tools/cacheSimulator/cacheSim: tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o
tools/cacheSimulator/cacheSim: tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o
tools/cacheSimulator/cacheSim: tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o
tools/cacheSimulator/cacheSim: tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o
tools/cacheSimulator/cacheSim: tools/cacheSimulator/CMakeFiles/cacheSim.dir/build.make
tools/cacheSimulator/cacheSim: tools/cacheSimulator/CMakeFiles/cacheSim.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable cacheSim"
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cacheSim.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/cacheSimulator/CMakeFiles/cacheSim.dir/build: tools/cacheSimulator/cacheSim
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/build

tools/cacheSimulator/CMakeFiles/cacheSim.dir/requires: tools/cacheSimulator/CMakeFiles/cacheSim.dir/stats.cpp.o.requires
tools/cacheSimulator/CMakeFiles/cacheSim.dir/requires: tools/cacheSimulator/CMakeFiles/cacheSim.dir/parse.cpp.o.requires
tools/cacheSimulator/CMakeFiles/cacheSim.dir/requires: tools/cacheSimulator/CMakeFiles/cacheSim.dir/cache.cpp.o.requires
tools/cacheSimulator/CMakeFiles/cacheSim.dir/requires: tools/cacheSimulator/CMakeFiles/cacheSim.dir/main.cpp.o.requires
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/requires

tools/cacheSimulator/CMakeFiles/cacheSim.dir/clean:
	cd /home/ewan/Desktop/visualizer/build/tools/cacheSimulator && $(CMAKE_COMMAND) -P CMakeFiles/cacheSim.dir/cmake_clean.cmake
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/clean

tools/cacheSimulator/CMakeFiles/cacheSim.dir/depend:
	cd /home/ewan/Desktop/visualizer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ewan/Desktop/visualizer /home/ewan/Desktop/visualizer/tools/cacheSimulator /home/ewan/Desktop/visualizer/build /home/ewan/Desktop/visualizer/build/tools/cacheSimulator /home/ewan/Desktop/visualizer/build/tools/cacheSimulator/CMakeFiles/cacheSim.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/cacheSimulator/CMakeFiles/cacheSim.dir/depend

