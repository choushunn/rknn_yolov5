#pragma once
#include "yolov5_detector.h"

class RKNNInferenceEngine {
private:
    std::vector<YoloV5Detector*> yolo_models;  // 存储多个 YOLO 模型对象的指针
    std::mutex mtx;  // 用于多线程访问的互斥锁

public:
    // 构造函数，初始化 YOLO 模型对象
    RKNNInferenceEngine(char* model_names, int num_core);

    // 析构函数，释放 YOLO 模型对象
    ~RKNNInferenceEngine();

    // 并行运行 YOLO 推理
    void run_inference(cv::Mat& image);
};