#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>

// 引入头文件
#include "rkdetect.h"

int main(int argc, char** argv)
{
	if (argc != 4)
	{
		printf("%s <model_path> <image_path>\n", argv[0]);
		return -1;
	}

	const char* model_path = argv[1];
	const char* image_path = argv[2];
	const char* type = argv[3];

	cv::VideoCapture cap;
	if (type == "cam") {
		cap.open(image_path); // 使用默认的摄像头
	}
	else if (type == "vid") {
		cap.open(image_path);
	}
	else {
		cap.open(image_path);
	}

	if (!cap.isOpened()) {
		printf("Failed to open");
		return -1;
	}

	auto start_time = std::chrono::high_resolution_clock::now();
	int frame_count = 0;

	cv::Mat source_img;
	// 输入图像
	image_buffer_t src_image;
	memset(&src_image, 0, sizeof(image_buffer_t));

	while (true) {
		frame_count++;
		auto current_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = current_time - start_time;
		double fps = frame_count / elapsed.count();
		// 读取一帧图像
		cap >> source_img;
		if (source_img.empty())
		{
			printf("无法加载图像: %s", image_path);
			return -1;
		}
		cv::Mat img;
		cv::cvtColor(source_img, img, cv::COLOR_BGR2RGB);
		cv::resize(img, img, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);

		src_image.width = img.cols;
		src_image.height = img.rows;
		src_image.format = IMAGE_FORMAT_RGB888;
		src_image.size = img.cols * img.rows * 3;
		src_image.virt_addr = img.data;
		// 保存检查结果
		object_detect_result_list od_results;
		// 调用算法执行检测
		int ret = DetectorAPI::run(&src_image, &od_results);

		if (ret != 0)
		{
			printf("rknn model fail! ret=%d\n", ret);
			return 0;
		}

		// 画框和概率
		char text[256];
		for (int i = 0; i < od_results.count; i++)
		{
			object_detect_result* det_result = &(od_results.results[i]);
			printf("%s @ (%d %d %d %d) %.3f\n", labels[det_result->cls_id],
				det_result->box.left, det_result->box.top,
				det_result->box.right, det_result->box.bottom,
				det_result->prop);
			int x1 = det_result->box.left;
			int y1 = det_result->box.top;
			int x2 = det_result->box.right;
			int y2 = det_result->box.bottom;
			// 在图像上绘制边框
			rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 4);
			sprintf(text, "%s %.1f%%", labels[det_result->cls_id], det_result->prop * 100);
			cv::Point org(x1, y1 - 10);
			putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(36, 255, 12), 2);
		}

		// 显示帧率
		std::string text1 = "FPS: " + std::to_string(fps);
		cv::putText(img, text1, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		// 显示结果
		cv::imshow("RKNN", img);

		// 按下 'q' 退出
		if (cv::waitKey(1) == 'q') {
			break;
		}
		if (type == "img") {
			cv::imwrite("out_opencv.jpg", img);
		}
	}

	// 释放Detector资源
	DetectorAPI::release();

	return 0;
}