#pragma once

class YoloV5Detector {
private:
    rknn_context rk_model;          // 指向 RKNN 模型的上下文指针
    unsigned char* model_data;      // 指向模型数据的指针
    rknn_sdk_version sdk_version;   // RKNN SDK 版本号
    rknn_input_output_num io_num;   // 输入输出数量
    rknn_tensor_attr* input_attrs;  // 输入张量属性数组
    rknn_tensor_attr* output_attrs; // 输出张量属性数组
    rknn_input inputs[1];           // 输入数据结构数组
    int ret;                // 返回值
    int channel_count = 3;          // 输入图像通道数
    int dst_width = 0;            // 输入图像宽度
    int dst_height = 0;           // 输入图像高度
public:
    cv::Mat original_image;         // 原始图像

    int pre_process();           // 预处理
    // 后处理
    int post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w,
        float conf_threshold, float nms_threshold, float scale_w, float scale_h,
        std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
        detect_result_group_t* group);
    // 成员函数：运行推理
    int run_inference(cv::Mat img);
    // 构造函数：初始化 YOLOv5 模型
    YoloV5Detector(char* model_name, int core_id);
    // 析构函数：释放资源
    ~YoloV5Detector();
};