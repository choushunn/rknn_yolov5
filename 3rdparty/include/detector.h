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
//#include "common.h"
#include "image_common.h"
#include "image_utils.h"
#include "file_utils.h"

#define OBJ_NAME_MAX_SIZE 64

#define OBJ_NUMB_MAX_SIZE 128

#define OBJ_CLASS_NUM 2

#define NMS_THRESH 0.45

#define BOX_THRESH 0.45

#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

typedef struct {
	image_rect_t box;
	float prop;
	int cls_id;
} object_detect_result;

// Ŀ�������б�
typedef struct {
	int id;
	int count;
	object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

// RKNN ������
typedef struct {
	// RKNN������
	rknn_context rknn_ctx;
	// ������������������
	rknn_input_output_num io_num;
	// ��������������
	rknn_tensor_attr* input_attrs;
	// �������������
	rknn_tensor_attr* output_attrs;
	// ģ�͵�ͨ����
	int model_channel;
	// ģ�͵Ŀ��
	int model_width;
	// ģ�͵ĸ߶�
	int model_height;
	// ��ʶģ���Ƿ�����
	bool is_quant;
} rknn_app_context_t;


static char* labels[OBJ_CLASS_NUM] = {
	"1",
	"2"
};

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
