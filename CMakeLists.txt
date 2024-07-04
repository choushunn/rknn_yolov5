cmake_minimum_required(VERSION 3.10)

Project(rknn_demo)
#set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/toolchain-rockchip-aarch64.cmake") 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")


#set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv410/lib/cmake/opencv4")
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv345/share/OpenCV")
find_package(OpenCV REQUIRED)
#set(mpp_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mpp")
#include_directories(${mpp_DIR}/include) 
#include_directories(${mpp_DIR}/include/rockchip) 
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/include") 

#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils) 
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
    #${mpp_DIR}/lib/librockit.so
    #${mpp_DIR}/lib/librockchip_mpp.so
    #${mpp_DIR}/lib/librockchip_vpu.so
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