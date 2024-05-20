cmake_minimum_required(VERSION 3.4.1)

project(rknn_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)


# 导入RKNN_API库
# 定义RKNN API库的路径
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/3rdparty/librknn_api)
# 定义RKNN运行时库的路径。
set(RKNN_RT_LIB ${RKNN_API_PATH}/lib/librknnrt.so)
# RKNN API的头文件
include_directories(${RKNN_API_PATH}/include)

# 导入rga库
set(RGA_PATH ${CMAKE_SOURCE_DIR}/3rdparty/rga)
set(RGA_LIB ${RGA_PATH}/lib/librga.so)
include_directories(${RGA_PATH}/include)

# 导入OpenCV库

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    find_package(OpenCV REQUIRED)
else()
    set(OpenCV_PATH ${CMAKE_SOURCE_DIR}/3rdparty/opencv)
    set(OpenCV_LIBS 
      ${OpenCV_PATH}/lib/libopencv_core.so
      ${OpenCV_PATH}/lib/libopencv_highgui.so
      ${OpenCV_PATH}/lib/libopencv_imgproc.so
      ${OpenCV_PATH}/lib/libopencv_videoio.so
      ${OpenCV_PATH}/lib/libopencv_imgcodecs.so
      ${OpenCV_PATH}/lib/libopencv_core.so.407
      ${OpenCV_PATH}/lib/libopencv_highgui.so.407
      ${OpenCV_PATH}/lib/libopencv_imgproc.so.407
      ${OpenCV_PATH}/lib/libopencv_videoio.so.407
      ${OpenCV_PATH}/lib/libopencv_imgcodecs.so.407
      ${OpenCV_PATH}/lib/libopencv_core.so.4.7.0
      ${OpenCV_PATH}/lib/libopencv_highgui.so.4.7.0
      ${OpenCV_PATH}/lib/libopencv_imgproc.so.4.7.0
      ${OpenCV_PATH}/lib/libopencv_videoio.so.4.7.0
      ${OpenCV_PATH}/lib/libopencv_imgcodecs.so.4.7.0
    )
    include_directories( ${OpenCV_PATH}/include/opencv4)
endif()

# 添加RKNNYoloV5子项目
add_subdirectory(RKNNYoloV5)

# rknn_demo
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/rknn_inference_engine.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
  rknnyolov5
)

# install target and libraries
install(TARGETS ${PROJECT_NAME} DESTINATION .)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(PROGRAMS ${OpenCV_LIBS} DESTINATION lib)
install(PROGRAMS RKNNYoloV5/lib/librknnyolov5.so DESTINATION lib)

install(DIRECTORY model DESTINATION .)
# 安装 run.sh 到当前目录
install(PROGRAMS run.sh DESTINATION .)