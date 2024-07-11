cmake_minimum_required(VERSION 3.10)

project(rknn_demo)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv410/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)


############################################################
# Add sub directories
############################################################
add_subdirectory(rkdetect)


############################################################
# Create an executable
############################################################
add_executable(${PROJECT_NAME} 
			   main.cpp
               )
#target_include_directories(${PROJECT_NAME} PRIVATE rkdetect/include)
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    rkdetect
    )

############################################################
# Install
############################################################
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# install target and libraries
install(TARGETS ${PROJECT_NAME} 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

#install(PROGRAMS ${OpenCV_LIBS} DESTINATION lib)
install(PROGRAMS rkdetect/lib/librkdetect.so DESTINATION lib)
# 安装 run.sh 到当前目录
install(PROGRAMS run.sh DESTINATION .)
install(DIRECTORY model DESTINATION .)
install(DIRECTORY data DESTINATION .)