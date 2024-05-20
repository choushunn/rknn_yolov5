#pragma once

#include <vector>
#include <thread>
#include <mutex>

#include "yolov5_detector.h"

class RKNNInferenceEngine {
private:
    std::vector<YoloV5Detector*> yolo_models;  // 存储多个 YOLO 模型对象的指针
    std::mutex mtx;  // 用于多线程访问的互斥锁

public:
    // 构造函数，初始化 YOLO 模型对象
    RKNNInferenceEngine(char* model_names, int num_core) {
        // 创建并初始化 YOLO 模型对象
        for (int i = 0; i < num_core; ++i) {
            int core_id = i % num_core;  // 循环使用核心
            YoloV5Detector* model = new YoloV5Detector(model_names, core_id);
            yolo_models.push_back(model);
        }
    }

    // 析构函数，释放 YOLO 模型对象
    ~RKNNInferenceEngine() {
        // 释放每个 YOLO 模型对象
        for (auto& model : yolo_models) {
            delete model;
        }
    }

    // 并行运行 YOLO 推理
    void run_inference(cv::Mat& image) {
        std::vector<std::thread> threads;

        // 启动每个 YOLO 模型的推理线程
        for (auto& model : yolo_models) {
            threads.push_back(std::thread(&YoloV5Detector::run_inference, model, std::ref(image)));
        }

        // 等待所有线程完成
        for (auto& thread : threads) {
            thread.join();
        }
    }
};