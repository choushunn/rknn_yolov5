#!/bin/bash
set -e

# 设置临时环境变量
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH

# 检查参数数量
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_path> <device_id>"
    exit 1
fi

# 获取参数
model_path="$1"
device_id="$2"
type_id="$3"
# 使用摄像头
./rknn_demo "$model_path" "$device_id" "$type_id"
