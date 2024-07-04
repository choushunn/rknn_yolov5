project(rkdetect)
project(${PROJECT_NAME} VERSION 1.0.1)
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty)

set(RKNN_API_PATH ${THIRD_PARTY_DIR}/librknn_api)
set(RGA_PATH ${THIRD_PARTY_DIR}/librga)
set(JPEG_TURBO_INCLUDE_DIR ${THIRD_PARTY_DIR}/jpeg_turbo/include)
set(STB_IMAGE_INCLUDE_DIR ${THIRD_PARTY_DIR}/stb_image/include)

set(RKNN_RT_LIB ${RKNN_API_PATH}/lib/librknnrt.so)
set(RGA_LIB ${RGA_PATH}/lib/librga.so)

add_library(${PROJECT_NAME}	SHARED
	src/detector.cpp
    src/rkdetect.cpp
	src/file_utils.c 
    src/image_drawing.c 
    src/image_utils.c
)

#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/include") 

target_include_directories(${PROJECT_NAME} PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
    ${RKNN_API_PATH}/include
    ${RGA_PATH}/include
    ${JPEG_TURBO_INCLUDE_DIR}
    ${STB_IMAGE_INCLUDE_DIR}
)

message(${RKNN_RT_LIB})
target_link_libraries(${PROJECT_NAME}
  ${RKNN_RT_LIB}
  ${RGA_LIB}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)