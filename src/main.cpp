#include <iostream>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "rknn_inference_engine.h"

using namespace std;
using namespace cv;

// 函数：检查字符串是否仅由数字字符组成
bool isNumeric(const string& str) {
    return !str.empty() && str.find_first_not_of("0123456789") == string::npos;
}

// 函数：检查字符串是否以特定前缀开头
bool startsWith(const string& str, const string& prefix) {
    return str.compare(0, prefix.size(), prefix) == 0;
}

// 初始化相机
bool init_camera(VideoCapture& capture, const string& device_path) {
	// 加载摄像头 CAP_ANY/CAP_V4L
    capture.open(device_path, cv::CAP_V4L);
    // capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('N', 'V', '1', '2'));
    if (!capture.isOpened()) {
        cerr << "Error: Failed to open the camera" << endl;
        return false;
    }
    return true;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <model_path> <image_name>" << endl;
        return -1;
    }

    char* model_path = argv[1];  // 参数二，模型所在路径
    const char* input_path = argv[2];  // 参数三, 视频/摄像头

    cout << "Model name: " << model_path << endl;

    // 初始化RKNN推理引擎
    int num_cores = 3; // NPU 核心，RK3588有三个核心
    RKNNInferenceEngine inference_engine(model_path, num_cores);
    VideoCapture capture;
	if (isNumeric(input_path) || startsWith(input_path, "/dev/video")) {
	   // 初始化相机
		if (!init_camera(capture, input_path)) {
			return -1;
		}
	}else{
		// 初始化视频文件
        capture.open(input_path);
        if (!capture.isOpened()) {
            cerr << "Error: Failed to open the video file" << endl;
            return -1;
        }
	}
     // 计算帧率
    int frame_count = 0;
    double total_time = 0.0;

    // 主循环
    Mat frame;
    while (true) {
        double start_time = static_cast<double>(getTickCount()); // 记录开始时间
        
        capture >> frame;
        if (frame.empty()) {
            cerr << "Error: Failed to capture frame" << endl;
            break;
        }
        inference_engine.run_inference(frame); 
        imshow("RKNN Yolo V5", frame);

        char key = waitKey(1);
        if (key == 27) // ESC key
            break;
    }

    capture.release();
    destroyAllWindows();
    return 0;
}
