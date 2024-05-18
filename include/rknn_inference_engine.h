#pragma once
#include "yolov5_detector.h"

class RKNNInferenceEngine {
private:
    std::vector<YoloV5Detector*> yolo_models;  // �洢��� YOLO ģ�Ͷ����ָ��
    std::mutex mtx;  // ���ڶ��̷߳��ʵĻ�����

public:
    // ���캯������ʼ�� YOLO ģ�Ͷ���
    RKNNInferenceEngine(char* model_names, int num_core);

    // �����������ͷ� YOLO ģ�Ͷ���
    ~RKNNInferenceEngine();

    // �������� YOLO ����
    void run_inference(cv::Mat& image);
};