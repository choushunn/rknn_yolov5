#pragma once
#include <iostream>
#include <set>
// OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
// rknn
#include "rknn_api.h"
// RGA
#include "im2d.hpp"
#include "RgaUtils.h"
#include "rga.h"


// 定义常量
// 最大目标名称长度
#define OBJ_NAME_MAX_SIZE 16
// 最大目标数量
#define OBJ_NUMB_MAX_SIZE 64
// 目标类别数量
#define OBJ_CLASS_NUM     4
// NMS 阈值
#define NMS_THRESH        0.45
// BOX 阈值
#define BOX_THRESH        0.25
// 单个目标属性大小
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

#define LABEL_NALE_TXT_PATH "./model/labels_list.txt"

static char* labels[OBJ_CLASS_NUM];

const int anchor0[6] = { 10, 13, 16, 30, 33, 23 };
const int anchor1[6] = { 30, 61, 62, 45, 59, 119 };
const int anchor2[6] = { 116, 90, 156, 198, 373, 326 };

inline static int clamp(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

// 定义矩形框结构体
typedef struct _BOX_RECT {
    int left;    // 左上角 x 坐标
    int right;   // 右下角 x 坐标
    int top;     // 左上角 y 坐标
    int bottom;  // 右下角 y 坐标
} BOX_RECT;

// 定义检测结果结构体
typedef struct __detect_result_t {
    char name[OBJ_NAME_MAX_SIZE];  // 目标名称
    BOX_RECT box;                   // 目标框
    float prop;                     // 目标置信度
} detect_result_t;

// 定义检测结果组结构体
typedef struct _detect_result_group_t {
    int id;                         // 组标识
    int count;                      // 目标数量
    detect_result_t results[OBJ_NUMB_MAX_SIZE];  // 目标检测结果数组
} detect_result_group_t;

class YoloV5Detector {
private:
    rknn_context rk_model;          // 指向 RKNN 模型的上下文指针
    unsigned char* model_data;      // 指向模型数据的指针
    rknn_sdk_version sdk_version;   // RKNN SDK 版本号
    rknn_input_output_num io_num;   // 输入输出数量
    rknn_tensor_attr* input_attrs;  // 输入张量属性数组
    rknn_tensor_attr* output_attrs; // 输出张量属性数组
    rknn_input inputs[1];           // 输入数据结构数组
    int ret;                // 返回值
    int channel_count = 3;          // 输入图像通道数
    int dst_width = 0;            // 输入图像宽度
    int dst_height = 0;           // 输入图像高度
public:
    cv::Mat original_image;         // 原始图像

    int pre_process();           // 预处理
    // 后处理
    int post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w,
        float conf_threshold, float nms_threshold, float scale_w, float scale_h,
        std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
        detect_result_group_t* group);
    // 成员函数：运行推理
    int run_inference(cv::Mat img);
    // 构造函数：初始化 YOLOv5 模型
    YoloV5Detector(const char* model_name, int core_id);
    // 析构函数：释放资源
    ~YoloV5Detector();
};