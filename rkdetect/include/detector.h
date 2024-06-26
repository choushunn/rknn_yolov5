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


// ���峣��
// ���Ŀ�����Ƴ���
#define OBJ_NAME_MAX_SIZE 16
// ���Ŀ������
#define OBJ_NUMB_MAX_SIZE 64
// Ŀ���������
#define OBJ_CLASS_NUM     4
// NMS ��ֵ
#define NMS_THRESH        0.45
// BOX ��ֵ
#define BOX_THRESH        0.25
// ����Ŀ�����Դ�С
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

#define LABEL_NALE_TXT_PATH "./model/labels_list.txt"

static char* labels[OBJ_CLASS_NUM];

const int anchor0[6] = { 10, 13, 16, 30, 33, 23 };
const int anchor1[6] = { 30, 61, 62, 45, 59, 119 };
const int anchor2[6] = { 116, 90, 156, 198, 373, 326 };

inline static int clamp(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

// ������ο�ṹ��
typedef struct _BOX_RECT {
    int left;    // ���Ͻ� x ����
    int right;   // ���½� x ����
    int top;     // ���Ͻ� y ����
    int bottom;  // ���½� y ����
} BOX_RECT;

// ���������ṹ��
typedef struct __detect_result_t {
    char name[OBJ_NAME_MAX_SIZE];  // Ŀ������
    BOX_RECT box;                   // Ŀ���
    float prop;                     // Ŀ�����Ŷ�
} detect_result_t;

// ����������ṹ��
typedef struct _detect_result_group_t {
    int id;                         // ���ʶ
    int count;                      // Ŀ������
    detect_result_t results[OBJ_NUMB_MAX_SIZE];  // Ŀ����������
} detect_result_group_t;

class YoloV5Detector {
private:
    rknn_context rk_model;          // ָ�� RKNN ģ�͵�������ָ��
    unsigned char* model_data;      // ָ��ģ�����ݵ�ָ��
    rknn_sdk_version sdk_version;   // RKNN SDK �汾��
    rknn_input_output_num io_num;   // �����������
    rknn_tensor_attr* input_attrs;  // ����������������
    rknn_tensor_attr* output_attrs; // ���������������
    rknn_input inputs[1];           // �������ݽṹ����
    int ret;                // ����ֵ
    int channel_count = 3;          // ����ͼ��ͨ����
    int dst_width = 0;            // ����ͼ����
    int dst_height = 0;           // ����ͼ��߶�
public:
    cv::Mat original_image;         // ԭʼͼ��

    int pre_process();           // Ԥ����
    // ����
    int post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w,
        float conf_threshold, float nms_threshold, float scale_w, float scale_h,
        std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
        detect_result_group_t* group);
    // ��Ա��������������
    int run_inference(cv::Mat img);
    // ���캯������ʼ�� YOLOv5 ģ��
    YoloV5Detector(const char* model_name, int core_id);
    // �����������ͷ���Դ
    ~YoloV5Detector();
};