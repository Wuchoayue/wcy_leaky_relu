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
include op_proto/CMakeFiles/cust_op_proto.dir/depend.make

# Include the progress variables for this target.
include op_proto/CMakeFiles/cust_op_proto.dir/progress.make

# Include the compile flags for this target's objects.
include op_proto/CMakeFiles/cust_op_proto.dir/flags.make

op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o: op_proto/CMakeFiles/cust_op_proto.dir/flags.make
op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o: ../op_proto/abs_npu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wu/wcy_leaky_relu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o"
	cd /home/wu/wcy_leaky_relu/build/op_proto && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o -c /home/wu/wcy_leaky_relu/op_proto/abs_npu.cpp

op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_op_proto.dir/abs_npu.cpp.i"
	cd /home/wu/wcy_leaky_relu/build/op_proto && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wu/wcy_leaky_relu/op_proto/abs_npu.cpp > CMakeFiles/cust_op_proto.dir/abs_npu.cpp.i

op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_op_proto.dir/abs_npu.cpp.s"
	cd /home/wu/wcy_leaky_relu/build/op_proto && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wu/wcy_leaky_relu/op_proto/abs_npu.cpp -o CMakeFiles/cust_op_proto.dir/abs_npu.cpp.s

op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o.requires:

.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o.requires

op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o.provides: op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o.requires
	$(MAKE) -f op_proto/CMakeFiles/cust_op_proto.dir/build.make op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o.provides.build
.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o.provides

op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o.provides.build: op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o


op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o: op_proto/CMakeFiles/cust_op_proto.dir/flags.make
op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o: ../op_proto/leaky_relu_npu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wu/wcy_leaky_relu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o"
	cd /home/wu/wcy_leaky_relu/build/op_proto && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o -c /home/wu/wcy_leaky_relu/op_proto/leaky_relu_npu.cpp

op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.i"
	cd /home/wu/wcy_leaky_relu/build/op_proto && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wu/wcy_leaky_relu/op_proto/leaky_relu_npu.cpp > CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.i

op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.s"
	cd /home/wu/wcy_leaky_relu/build/op_proto && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wu/wcy_leaky_relu/op_proto/leaky_relu_npu.cpp -o CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.s

op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o.requires:

.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o.requires

op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o.provides: op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o.requires
	$(MAKE) -f op_proto/CMakeFiles/cust_op_proto.dir/build.make op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o.provides.build
.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o.provides

op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o.provides.build: op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o


op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o: op_proto/CMakeFiles/cust_op_proto.dir/flags.make
op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o: ../op_proto/wu_relu_npu.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wu/wcy_leaky_relu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o"
	cd /home/wu/wcy_leaky_relu/build/op_proto && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o -c /home/wu/wcy_leaky_relu/op_proto/wu_relu_npu.cpp

op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.i"
	cd /home/wu/wcy_leaky_relu/build/op_proto && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wu/wcy_leaky_relu/op_proto/wu_relu_npu.cpp > CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.i

op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.s"
	cd /home/wu/wcy_leaky_relu/build/op_proto && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wu/wcy_leaky_relu/op_proto/wu_relu_npu.cpp -o CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.s

op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o.requires:

.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o.requires

op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o.provides: op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o.requires
	$(MAKE) -f op_proto/CMakeFiles/cust_op_proto.dir/build.make op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o.provides.build
.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o.provides

op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o.provides.build: op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o


# Object files for target cust_op_proto
cust_op_proto_OBJECTS = \
"CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o" \
"CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o" \
"CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o"

# External object files for target cust_op_proto
cust_op_proto_EXTERNAL_OBJECTS =

makepkg/packages/op_proto/custom/libcust_op_proto.so: op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o
makepkg/packages/op_proto/custom/libcust_op_proto.so: op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o
makepkg/packages/op_proto/custom/libcust_op_proto.so: op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o
makepkg/packages/op_proto/custom/libcust_op_proto.so: op_proto/CMakeFiles/cust_op_proto.dir/build.make
makepkg/packages/op_proto/custom/libcust_op_proto.so: op_proto/CMakeFiles/cust_op_proto.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wu/wcy_leaky_relu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library ../makepkg/packages/op_proto/custom/libcust_op_proto.so"
	cd /home/wu/wcy_leaky_relu/build/op_proto && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cust_op_proto.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
op_proto/CMakeFiles/cust_op_proto.dir/build: makepkg/packages/op_proto/custom/libcust_op_proto.so

.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/build

op_proto/CMakeFiles/cust_op_proto.dir/requires: op_proto/CMakeFiles/cust_op_proto.dir/abs_npu.cpp.o.requires
op_proto/CMakeFiles/cust_op_proto.dir/requires: op_proto/CMakeFiles/cust_op_proto.dir/leaky_relu_npu.cpp.o.requires
op_proto/CMakeFiles/cust_op_proto.dir/requires: op_proto/CMakeFiles/cust_op_proto.dir/wu_relu_npu.cpp.o.requires

.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/requires

op_proto/CMakeFiles/cust_op_proto.dir/clean:
	cd /home/wu/wcy_leaky_relu/build/op_proto && $(CMAKE_COMMAND) -P CMakeFiles/cust_op_proto.dir/cmake_clean.cmake
.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/clean

op_proto/CMakeFiles/cust_op_proto.dir/depend:
	cd /home/wu/wcy_leaky_relu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wu/wcy_leaky_relu /home/wu/wcy_leaky_relu/op_proto /home/wu/wcy_leaky_relu/build /home/wu/wcy_leaky_relu/build/op_proto /home/wu/wcy_leaky_relu/build/op_proto/CMakeFiles/cust_op_proto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/depend

