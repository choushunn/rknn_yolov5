#pragma once

#include <vector>
#include <thread>
#include <mutex>

#include "yolov5_detector.h"

class RKNNInferenceEngine {
private:
    std::vector<YoloV5Detector*> yolo_models;  // �洢��� YOLO ģ�Ͷ����ָ��
    std::mutex mtx;  // ���ڶ��̷߳��ʵĻ�����

public:
    // ���캯������ʼ�� YOLO ģ�Ͷ���
    RKNNInferenceEngine(char* model_names, int num_core) {
        // ��������ʼ�� YOLO ģ�Ͷ���
        for (int i = 0; i < num_core; ++i) {
            int core_id = i % num_core;  // ѭ��ʹ�ú���
            YoloV5Detector* model = new YoloV5Detector(model_names, core_id);
            yolo_models.push_back(model);
        }
    }

    // �����������ͷ� YOLO ģ�Ͷ���
    ~RKNNInferenceEngine() {
        // �ͷ�ÿ�� YOLO ģ�Ͷ���
        for (auto& model : yolo_models) {
            delete model;
        }
    }

    // �������� YOLO ����
    void run_inference(cv::Mat& image) {
        std::vector<std::thread> threads;

        // ����ÿ�� YOLO ģ�͵������߳�
        for (auto& model : yolo_models) {
            threads.push_back(std::thread(&YoloV5Detector::run_inference, model, std::ref(image)));
        }

        // �ȴ������߳����
        for (auto& thread : threads) {
            thread.join();
        }
    }
};