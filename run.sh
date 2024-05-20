#!/bin/bash
set -e

# 设置临时环境变量
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH


# 使用摄像头
./rknn_demo ./model/yolov5s-640-640.rknn 0

