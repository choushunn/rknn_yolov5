#include "yolov5_detector.h"

int YoloV5Detector::pre_process()
{
	return 0;
}

int YoloV5Detector::post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales, detect_result_group_t* group)
{
	return 0;
}

int YoloV5Detector::run_inference(cv::Mat img)
{
	return 0;
}

YoloV5Detector::YoloV5Detector(char* model_name, int core_id)
{
}

YoloV5Detector::~YoloV5Detector()
{

}
