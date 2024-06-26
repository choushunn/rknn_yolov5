# RKNN_YOLOV5

> 实现多线程并行运行YoloV5

## 依赖

- [RKNN](https://github.com/airockchip/rknn-toolkit2)  # RKNPU2 版本与模型转换版本需要一致
- [RGA](https://github.com/airockchip/librga)
- [OpenCV](https://opencv.org/releases/)
- 交叉编译工具链

## 编译

```bash
sudo ./build-linux_RK3588.sh
```

## 运行

```bash
cd install

sudo ./run.sh model/yolov5s.rknn /dev/video21
# or
sudo ./run.sh model/yolov5s.rknn ./data/test.mp4
```

```
# 设置临时环境变量
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
```