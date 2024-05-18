#include <vector>
#include <thread>
#include <mutex>

#include "rknn_inference_engine.h"

RKNNInferenceEngine::RKNNInferenceEngine(char* model_names, int num_core)
{
	// ��������ʼ�� YOLO ģ�Ͷ���
	for (int i = 0; i < num_core; ++i) {
		int core_id = i % num_core;  // ѭ��ʹ�ú���
		RKNNYoloV5* model = new RKNNYoloV5(model_names, core_id);
		yolo_models.push_back(model);
	}
}

void RKNNInferenceEngine::run_inference(cv::Mat& image)
{
	std::vector<std::thread> threads;

	// ����ÿ�� YOLO ģ�͵������߳�
	for (auto& model : yolo_models) {
		threads.push_back(std::thread(&RKNNYoloV5::run_inference, model, std::ref(image)));
	}

	// �ȴ������߳����
	for (auto& thread : threads) {
		thread.join();
	}
}



RKNNInferenceEngine::~RKNNInferenceEngine()
{
	// �ͷ�ÿ�� YOLO ģ�Ͷ���
	for (auto& model : yolo_models) {
		delete model;
	}
}