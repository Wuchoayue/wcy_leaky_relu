# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/wu/wcy_leaky_relu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wu/wcy_leaky_relu/build

# Include any dependencies generated for this target.
include framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/depend.make

# Include the progress variables for this target.
include framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/progress.make

# Include the compile flags for this target's objects.
include framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/flags.make

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/flags.make
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o: ../framework/onnx_plugin/abs_npu_plugin.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wu/wcy_leaky_relu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o"
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o -c /home/wu/wcy_leaky_relu/framework/onnx_plugin/abs_npu_plugin.cpp

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.i"
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wu/wcy_leaky_relu/framework/onnx_plugin/abs_npu_plugin.cpp > CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.i

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.s"
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wu/wcy_leaky_relu/framework/onnx_plugin/abs_npu_plugin.cpp -o CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.s

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o.requires:

.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o.requires

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o.provides: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o.requires
	$(MAKE) -f framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/build.make framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o.provides.build
.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o.provides

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o.provides.build: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o


framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/flags.make
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o: ../framework/onnx_plugin/leaky_relu_npu_plugin.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wu/wcy_leaky_relu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o"
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o -c /home/wu/wcy_leaky_relu/framework/onnx_plugin/leaky_relu_npu_plugin.cpp

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.i"
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wu/wcy_leaky_relu/framework/onnx_plugin/leaky_relu_npu_plugin.cpp > CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.i

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.s"
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wu/wcy_leaky_relu/framework/onnx_plugin/leaky_relu_npu_plugin.cpp -o CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.s

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o.requires:

.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o.requires

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o.provides: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o.requires
	$(MAKE) -f framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/build.make framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o.provides.build
.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o.provides

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o.provides.build: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o


framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/flags.make
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o: ../framework/onnx_plugin/wu_npu_plugin.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wu/wcy_leaky_relu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o"
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o -c /home/wu/wcy_leaky_relu/framework/onnx_plugin/wu_npu_plugin.cpp

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.i"
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wu/wcy_leaky_relu/framework/onnx_plugin/wu_npu_plugin.cpp > CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.i

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.s"
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wu/wcy_leaky_relu/framework/onnx_plugin/wu_npu_plugin.cpp -o CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.s

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o.requires:

.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o.requires

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o.provides: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o.requires
	$(MAKE) -f framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/build.make framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o.provides.build
.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o.provides

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o.provides.build: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o


# Object files for target cust_onnx_parsers
cust_onnx_parsers_OBJECTS = \
"CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o" \
"CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o" \
"CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o"

# External object files for target cust_onnx_parsers
cust_onnx_parsers_EXTERNAL_OBJECTS =

makepkg/packages/framework/custom/onnx/libcust_onnx_parsers.so: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o
makepkg/packages/framework/custom/onnx/libcust_onnx_parsers.so: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o
makepkg/packages/framework/custom/onnx/libcust_onnx_parsers.so: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o
makepkg/packages/framework/custom/onnx/libcust_onnx_parsers.so: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/build.make
makepkg/packages/framework/custom/onnx/libcust_onnx_parsers.so: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wu/wcy_leaky_relu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library ../../makepkg/packages/framework/custom/onnx/libcust_onnx_parsers.so"
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cust_onnx_parsers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/build: makepkg/packages/framework/custom/onnx/libcust_onnx_parsers.so

.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/build

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/requires: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/abs_npu_plugin.cpp.o.requires
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/requires: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_npu_plugin.cpp.o.requires
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/requires: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/wu_npu_plugin.cpp.o.requires

.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/requires

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/clean:
	cd /home/wu/wcy_leaky_relu/build/framework/onnx_plugin && $(CMAKE_COMMAND) -P CMakeFiles/cust_onnx_parsers.dir/cmake_clean.cmake
.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/clean

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/depend:
	cd /home/wu/wcy_leaky_relu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wu/wcy_leaky_relu /home/wu/wcy_leaky_relu/framework/onnx_plugin /home/wu/wcy_leaky_relu/build /home/wu/wcy_leaky_relu/build/framework/onnx_plugin /home/wu/wcy_leaky_relu/build/framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/depend

