# 声明这个目录下的源文件
set(SOURCE_FILES
    utils.cpp
    yolo_utils.cpp
    rknn_utils.cpp
    # 其他工具源文件
)

# 添加头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# 添加静态库/动态库的构建方式，这里以静态库为例
add_library(utils STATIC ${SOURCE_FILES})

# 设置编译选项等
# ...

# 如果需要链接其他库，可以在这里添加
# target_link_libraries(utils ${SOME_OTHER_LIBRARIES})
