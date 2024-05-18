#pragma once

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
    YoloV5Detector(char* model_name, int core_id);
    // �����������ͷ���Դ
    ~YoloV5Detector();
};