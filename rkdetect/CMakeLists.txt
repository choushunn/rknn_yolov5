project(rkdetect VERSION 1.0.1)

set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/3rdparty)

add_library(${PROJECT_NAME}	SHARED
	src/file_utils.c 
    src/image_utils.c
	src/detector.cpp
    src/rkdetect.cpp
)

set(RKNN_INCLUDE_DIR ${THIRD_PARTY_DIR}/librknn_api/include)
set(RGA_INCLUDE_DIR ${THIRD_PARTY_DIR}/librga/include)
set(JPEG_TURBO_INCLUDE_DIR ${THIRD_PARTY_DIR}/jpeg_turbo/include)
set(STB_IMAGE_INCLUDE_DIR ${THIRD_PARTY_DIR}/stb_image/include)
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(${PROJECT_NAME} PUBLIC
	${PROJECT_INCLUDE_DIR}
    ${RKNN_INCLUDE_DIR}
    ${RGA_INCLUDE_DIR}
    ${JPEG_TURBO_INCLUDE_DIR}
    ${STB_IMAGE_INCLUDE_DIR}
)

set(RKNN_RT_LIB ${THIRD_PARTY_DIR}/librknn_api/lib/librknnrt.so)
set(RGA_LIB ${THIRD_PARTY_DIR}/librga/lib/librga.so)

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${RKNN_RT_LIB}
  ${RGA_LIB}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)