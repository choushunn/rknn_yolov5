# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\MyGithub\rknn_detect

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\MyGithub\rknn_detect\build

# Include any dependencies generated for this target.
include rkdetect/CMakeFiles/rkdetect.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include rkdetect/CMakeFiles/rkdetect.dir/compiler_depend.make

# Include the progress variables for this target.
include rkdetect/CMakeFiles/rkdetect.dir/progress.make

# Include the compile flags for this target's objects.
include rkdetect/CMakeFiles/rkdetect.dir/flags.make

rkdetect/CMakeFiles/rkdetect.dir/src/detector.cpp.obj: rkdetect/CMakeFiles/rkdetect.dir/flags.make
rkdetect/CMakeFiles/rkdetect.dir/src/detector.cpp.obj: rkdetect/CMakeFiles/rkdetect.dir/includes_CXX.rsp
rkdetect/CMakeFiles/rkdetect.dir/src/detector.cpp.obj: D:/MyGithub/rknn_detect/rkdetect/src/detector.cpp
rkdetect/CMakeFiles/rkdetect.dir/src/detector.cpp.obj: rkdetect/CMakeFiles/rkdetect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=D:\MyGithub\rknn_detect\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object rkdetect/CMakeFiles/rkdetect.dir/src/detector.cpp.obj"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT rkdetect/CMakeFiles/rkdetect.dir/src/detector.cpp.obj -MF CMakeFiles\rkdetect.dir\src\detector.cpp.obj.d -o CMakeFiles\rkdetect.dir\src\detector.cpp.obj -c D:\MyGithub\rknn_detect\rkdetect\src\detector.cpp

rkdetect/CMakeFiles/rkdetect.dir/src/detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/rkdetect.dir/src/detector.cpp.i"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\MyGithub\rknn_detect\rkdetect\src\detector.cpp > CMakeFiles\rkdetect.dir\src\detector.cpp.i

rkdetect/CMakeFiles/rkdetect.dir/src/detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/rkdetect.dir/src/detector.cpp.s"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\MyGithub\rknn_detect\rkdetect\src\detector.cpp -o CMakeFiles\rkdetect.dir\src\detector.cpp.s

rkdetect/CMakeFiles/rkdetect.dir/src/rkdetect.cpp.obj: rkdetect/CMakeFiles/rkdetect.dir/flags.make
rkdetect/CMakeFiles/rkdetect.dir/src/rkdetect.cpp.obj: rkdetect/CMakeFiles/rkdetect.dir/includes_CXX.rsp
rkdetect/CMakeFiles/rkdetect.dir/src/rkdetect.cpp.obj: D:/MyGithub/rknn_detect/rkdetect/src/rkdetect.cpp
rkdetect/CMakeFiles/rkdetect.dir/src/rkdetect.cpp.obj: rkdetect/CMakeFiles/rkdetect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=D:\MyGithub\rknn_detect\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object rkdetect/CMakeFiles/rkdetect.dir/src/rkdetect.cpp.obj"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT rkdetect/CMakeFiles/rkdetect.dir/src/rkdetect.cpp.obj -MF CMakeFiles\rkdetect.dir\src\rkdetect.cpp.obj.d -o CMakeFiles\rkdetect.dir\src\rkdetect.cpp.obj -c D:\MyGithub\rknn_detect\rkdetect\src\rkdetect.cpp

rkdetect/CMakeFiles/rkdetect.dir/src/rkdetect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/rkdetect.dir/src/rkdetect.cpp.i"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\MyGithub\rknn_detect\rkdetect\src\rkdetect.cpp > CMakeFiles\rkdetect.dir\src\rkdetect.cpp.i

rkdetect/CMakeFiles/rkdetect.dir/src/rkdetect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/rkdetect.dir/src/rkdetect.cpp.s"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\MyGithub\rknn_detect\rkdetect\src\rkdetect.cpp -o CMakeFiles\rkdetect.dir\src\rkdetect.cpp.s

rkdetect/CMakeFiles/rkdetect.dir/src/file_utils.c.obj: rkdetect/CMakeFiles/rkdetect.dir/flags.make
rkdetect/CMakeFiles/rkdetect.dir/src/file_utils.c.obj: rkdetect/CMakeFiles/rkdetect.dir/includes_C.rsp
rkdetect/CMakeFiles/rkdetect.dir/src/file_utils.c.obj: D:/MyGithub/rknn_detect/rkdetect/src/file_utils.c
rkdetect/CMakeFiles/rkdetect.dir/src/file_utils.c.obj: rkdetect/CMakeFiles/rkdetect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=D:\MyGithub\rknn_detect\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object rkdetect/CMakeFiles/rkdetect.dir/src/file_utils.c.obj"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT rkdetect/CMakeFiles/rkdetect.dir/src/file_utils.c.obj -MF CMakeFiles\rkdetect.dir\src\file_utils.c.obj.d -o CMakeFiles\rkdetect.dir\src\file_utils.c.obj -c D:\MyGithub\rknn_detect\rkdetect\src\file_utils.c

rkdetect/CMakeFiles/rkdetect.dir/src/file_utils.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/rkdetect.dir/src/file_utils.c.i"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E D:\MyGithub\rknn_detect\rkdetect\src\file_utils.c > CMakeFiles\rkdetect.dir\src\file_utils.c.i

rkdetect/CMakeFiles/rkdetect.dir/src/file_utils.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/rkdetect.dir/src/file_utils.c.s"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S D:\MyGithub\rknn_detect\rkdetect\src\file_utils.c -o CMakeFiles\rkdetect.dir\src\file_utils.c.s

rkdetect/CMakeFiles/rkdetect.dir/src/image_drawing.c.obj: rkdetect/CMakeFiles/rkdetect.dir/flags.make
rkdetect/CMakeFiles/rkdetect.dir/src/image_drawing.c.obj: rkdetect/CMakeFiles/rkdetect.dir/includes_C.rsp
rkdetect/CMakeFiles/rkdetect.dir/src/image_drawing.c.obj: D:/MyGithub/rknn_detect/rkdetect/src/image_drawing.c
rkdetect/CMakeFiles/rkdetect.dir/src/image_drawing.c.obj: rkdetect/CMakeFiles/rkdetect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=D:\MyGithub\rknn_detect\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object rkdetect/CMakeFiles/rkdetect.dir/src/image_drawing.c.obj"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT rkdetect/CMakeFiles/rkdetect.dir/src/image_drawing.c.obj -MF CMakeFiles\rkdetect.dir\src\image_drawing.c.obj.d -o CMakeFiles\rkdetect.dir\src\image_drawing.c.obj -c D:\MyGithub\rknn_detect\rkdetect\src\image_drawing.c

rkdetect/CMakeFiles/rkdetect.dir/src/image_drawing.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/rkdetect.dir/src/image_drawing.c.i"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E D:\MyGithub\rknn_detect\rkdetect\src\image_drawing.c > CMakeFiles\rkdetect.dir\src\image_drawing.c.i

rkdetect/CMakeFiles/rkdetect.dir/src/image_drawing.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/rkdetect.dir/src/image_drawing.c.s"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S D:\MyGithub\rknn_detect\rkdetect\src\image_drawing.c -o CMakeFiles\rkdetect.dir\src\image_drawing.c.s

rkdetect/CMakeFiles/rkdetect.dir/src/image_utils.c.obj: rkdetect/CMakeFiles/rkdetect.dir/flags.make
rkdetect/CMakeFiles/rkdetect.dir/src/image_utils.c.obj: rkdetect/CMakeFiles/rkdetect.dir/includes_C.rsp
rkdetect/CMakeFiles/rkdetect.dir/src/image_utils.c.obj: D:/MyGithub/rknn_detect/rkdetect/src/image_utils.c
rkdetect/CMakeFiles/rkdetect.dir/src/image_utils.c.obj: rkdetect/CMakeFiles/rkdetect.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=D:\MyGithub\rknn_detect\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object rkdetect/CMakeFiles/rkdetect.dir/src/image_utils.c.obj"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT rkdetect/CMakeFiles/rkdetect.dir/src/image_utils.c.obj -MF CMakeFiles\rkdetect.dir\src\image_utils.c.obj.d -o CMakeFiles\rkdetect.dir\src\image_utils.c.obj -c D:\MyGithub\rknn_detect\rkdetect\src\image_utils.c

rkdetect/CMakeFiles/rkdetect.dir/src/image_utils.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/rkdetect.dir/src/image_utils.c.i"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E D:\MyGithub\rknn_detect\rkdetect\src\image_utils.c > CMakeFiles\rkdetect.dir\src\image_utils.c.i

rkdetect/CMakeFiles/rkdetect.dir/src/image_utils.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/rkdetect.dir/src/image_utils.c.s"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && C:\Programs\Qt\Tools\mingw1120_64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S D:\MyGithub\rknn_detect\rkdetect\src\image_utils.c -o CMakeFiles\rkdetect.dir\src\image_utils.c.s

# Object files for target rkdetect
rkdetect_OBJECTS = \
"CMakeFiles/rkdetect.dir/src/detector.cpp.obj" \
"CMakeFiles/rkdetect.dir/src/rkdetect.cpp.obj" \
"CMakeFiles/rkdetect.dir/src/file_utils.c.obj" \
"CMakeFiles/rkdetect.dir/src/image_drawing.c.obj" \
"CMakeFiles/rkdetect.dir/src/image_utils.c.obj"

# External object files for target rkdetect
rkdetect_EXTERNAL_OBJECTS =

rkdetect/librkdetect.dll: rkdetect/CMakeFiles/rkdetect.dir/src/detector.cpp.obj
rkdetect/librkdetect.dll: rkdetect/CMakeFiles/rkdetect.dir/src/rkdetect.cpp.obj
rkdetect/librkdetect.dll: rkdetect/CMakeFiles/rkdetect.dir/src/file_utils.c.obj
rkdetect/librkdetect.dll: rkdetect/CMakeFiles/rkdetect.dir/src/image_drawing.c.obj
rkdetect/librkdetect.dll: rkdetect/CMakeFiles/rkdetect.dir/src/image_utils.c.obj
rkdetect/librkdetect.dll: rkdetect/CMakeFiles/rkdetect.dir/build.make
rkdetect/librkdetect.dll: D:/MyGithub/rknn_detect/rkdetect/../3rdparty/librknn_api/lib/librknnrt.so
rkdetect/librkdetect.dll: D:/MyGithub/rknn_detect/rkdetect/../3rdparty/librga/lib/librga.so
rkdetect/librkdetect.dll: rkdetect/CMakeFiles/rkdetect.dir/linkLibs.rsp
rkdetect/librkdetect.dll: rkdetect/CMakeFiles/rkdetect.dir/objects1.rsp
rkdetect/librkdetect.dll: rkdetect/CMakeFiles/rkdetect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=D:\MyGithub\rknn_detect\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX shared library librkdetect.dll"
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\rkdetect.dir\link.txt --verbose=$(VERBOSE)
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && "C:\Program Files\PowerShell\7\pwsh.exe" -noprofile -executionpolicy Bypass -file C:/Programs/vcpkg/scripts/buildsystems/msbuild/applocal.ps1 -targetBinary D:/MyGithub/rknn_detect/build/rkdetect/librkdetect.dll -installedDir C:/Programs/vcpkg/installed/x64-windows/debug/bin -OutVariable out

# Rule to build all files generated by this target.
rkdetect/CMakeFiles/rkdetect.dir/build: rkdetect/librkdetect.dll
.PHONY : rkdetect/CMakeFiles/rkdetect.dir/build

rkdetect/CMakeFiles/rkdetect.dir/clean:
	cd /d D:\MyGithub\rknn_detect\build\rkdetect && $(CMAKE_COMMAND) -P CMakeFiles\rkdetect.dir\cmake_clean.cmake
.PHONY : rkdetect/CMakeFiles/rkdetect.dir/clean

rkdetect/CMakeFiles/rkdetect.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\MyGithub\rknn_detect D:\MyGithub\rknn_detect\rkdetect D:\MyGithub\rknn_detect\build D:\MyGithub\rknn_detect\build\rkdetect D:\MyGithub\rknn_detect\build\rkdetect\CMakeFiles\rkdetect.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : rkdetect/CMakeFiles/rkdetect.dir/depend

