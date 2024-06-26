cmake_minimum_required(VERSION 3.4.1)

project(rkdetect)
project(${PROJECT_NAME} VERSION 1.0.1)
# RKNN API库的路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/lib/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)

# 导入rga库
set(RGA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/librga)
set(RGA_LIB ${RGA_PATH}/lib/librga.so)
include_directories(${RGA_PATH}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/jpeg_turbo/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/stb_image/include)

add_library(${PROJECT_NAME}	SHARED
	src/detector.cpp
	../../utils/file_utils.c 
    ../../utils/image_drawing.c
    ../../utils/image_utils.c
)

target_link_libraries(${PROJECT_NAME}
  ${RKNN_RT_LIB}
  ${RGA_LIB}
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)