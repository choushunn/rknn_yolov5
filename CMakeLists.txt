cmake_minimum_required(VERSION 3.10)

Project(rknn_demo)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")


set(OpenCV_DIR "3rdparty/opencv345/share/OpenCV")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librknn_api/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rkdetect/include)

# 添加子项目
add_subdirectory(rkdetect)

add_executable(${PROJECT_NAME} 
			   main.cpp
               )

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    rkdetect
    )

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# install target and libraries
install(TARGETS ${PROJECT_NAME} DESTINATION .)

#install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
#install(PROGRAMS ${RGA_LIB} DESTINATION lib)
#install(PROGRAMS ${OpenCV_LIBS} DESTINATION lib)
install(PROGRAMS rkdetect/lib/librkdetect.so DESTINATION lib)
install(DIRECTORY model DESTINATION .)
install(DIRECTORY data DESTINATION .)
# 安装 run.sh 到当前目录
install(PROGRAMS run.sh DESTINATION .)