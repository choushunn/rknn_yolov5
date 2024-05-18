#include <vector>
#include <thread>
#include <mutex>

#include "rknn_inference_engine.h"

RKNNInferenceEngine::RKNNInferenceEngine(char* model_names, int num_core)
{
	// 创建并初始化 YOLO 模型对象
	for (int i = 0; i < num_core; ++i) {
		int core_id = i % num_core;  // 循环使用核心
		RKNNYoloV5* model = new RKNNYoloV5(model_names, core_id);
		yolo_models.push_back(model);
	}
}

void RKNNInferenceEngine::run_inference(cv::Mat& image)
{
	std::vector<std::thread> threads;

	// 启动每个 YOLO 模型的推理线程
	for (auto& model : yolo_models) {
		threads.push_back(std::thread(&RKNNYoloV5::run_inference, model, std::ref(image)));
	}

	// 等待所有线程完成
	for (auto& thread : threads) {
		thread.join();
	}
}



RKNNInferenceEngine::~RKNNInferenceEngine()
{
	// 释放每个 YOLO 模型对象
	for (auto& model : yolo_models) {
		delete model;
	}
}