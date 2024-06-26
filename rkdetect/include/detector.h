#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <vector>
#include <set>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"
#include "file_utils.h"

// 定义目标名称的最大尺寸
#define OBJ_NAME_MAX_SIZE 64

// 定义目标数量的最大尺寸 
#define OBJ_NUMB_MAX_SIZE 128

// 定义目标类别的数量
#define OBJ_CLASS_NUM 4

// 定义非极大值抑制(NMS)的阈值
#define NMS_THRESH 0.45

// 定义边框置信度的阈值
#define BOX_THRESH 0.45

// 定义属性框的大小
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

// 目标检测操作的结果
typedef struct {
	// 检测到的目标的边界框
	image_rect_t box;
	// 检测的置信度得分
	float prop;
	// 检测到的目标的类别ID
	int cls_id;
} object_detect_result;

// 目标检测结果列表
typedef struct {
	// 检测结果列表的ID
	int id;
	// 检测到的目标数量
	int count;
	// 目标检测结果数组
	object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

// RKNN 上下文
typedef struct {
	// RKNN上下文
	rknn_context rknn_ctx;
	// 输入和输出张量的数量
	rknn_input_output_num io_num;
	// 输入张量的属性
	rknn_tensor_attr* input_attrs;
	// 输出张量的属性
	rknn_tensor_attr* output_attrs;
	// 模型的通道数
	int model_channel;
	// 模型的宽度
	int model_width;
	// 模型的高度
	int model_height;
	// 标识模型是否量化
	bool is_quant;
} rknn_app_context_t;

// 存储类别标签的数组
static char* labels[OBJ_CLASS_NUM] = {
	"1",
	"2",
	"3",
	"4"
};

// 锚框的尺寸和比例
const int anchor[3][6] = {
	{10, 13, 16, 30, 33, 23},
	{30, 61, 62, 45, 59, 119},
	{116, 90, 156, 198, 373, 326}
};

class Detector {
private:
	rknn_app_context_t* app_ctx;
	int post_process(void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, object_detect_result_list* od_results);
public:
	Detector(const char* model_path);
	~Detector();
	int run(image_buffer_t* img, object_detect_result_list* od_results);	
};

#endif //_DETECTOR_H_