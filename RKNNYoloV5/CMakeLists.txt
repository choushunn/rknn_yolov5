cmake_minimum_required(VERSION 3.4.1)

project(rknnyolov5)
project(${PROJECT_NAME} VERSION 1.0.0)
add_library(${PROJECT_NAME}	SHARED
	src/yolov5_detector.cpp
	src/yolov5_utils.cpp
)

# 导入RKNN_API库
# 定义RKNN API库的路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/librknn_api)
# 定义RKNN运行时库的路径。
set(RKNN_RT_LIB ${RKNN_API_PATH}/lib/librknnrt.so)
# RKNN API的头文件
include_directories(${RKNN_API_PATH}/include)

# 导入rga库
set(RGA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/rga)
set(RGA_LIB ${RGA_PATH}/lib/librga.so)
include_directories(${RGA_PATH}/include)

target_link_libraries(${PROJECT_NAME}
  ${RKNN_RT_LIB}
  ${RGA_LIB}
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)