#include "yolov5_detector.h"

int YoloV5Detector::pre_process()
{
	return 0;
}


/**
 * @brief 计算两个矩形框的重叠度
 *
 * @param xmin0 第一个矩形框的左上角 x 坐标
 * @param ymin0 第一个矩形框的左上角 y 坐标
 * @param xmax0 第一个矩形框的右下角 x 坐标
 * @param ymax0 第一个矩形框的右下角 y 坐标
 * @param xmin1 第二个矩形框的左上角 x 坐标
 * @param ymin1 第二个矩形框的左上角 y 坐标
 * @param xmax1 第二个矩形框的右下角 x 坐标
 * @param ymax1 第二个矩形框的右下角 y 坐标
 * @return float 重叠度（IOU）
 */
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
	float xmin1, float ymin1, float xmax1, float ymax1) {
	float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0); // 交集宽度
	float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0); // 交集高度
	float i = w * h; // 交集面积
	float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i; // 并集面积
	return u <= 0.f ? 0.f : (i / u); // 计算 IOU 并返回
}

/**
 * @brief NMS 非极大值抑制
 *
 * @param validCount 有效检测框数量
 * @param outputLocations 检测框位置信息数组
 * @param classIds 检测框类别数组
 * @param order 排序索引数组
 * @param filterId 需要过滤的类别 ID
 * @param threshold IOU 阈值
 * @return int 返回值为 0 表示成功
 */
static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds,
	std::vector<int>& order, int filterId, float threshold) {
	// 遍历所有检测框
	for (int i = 0; i < validCount; ++i) {
		// 如果当前检测框无效或类别不匹配，则跳过
		if (order[i] == -1 || classIds[i] != filterId) {
			continue;
		}
		int n = order[i];
		// 继续遍历后续检测框
		for (int j = i + 1; j < validCount; ++j) {
			int m = order[j];
			// 如果当前检测框无效或类别不匹配，则跳过
			if (m == -1 || classIds[i] != filterId) {
				continue;
			}
			// 获取当前和后续检测框的坐标信息
			float xmin0 = outputLocations[n * 4 + 0];
			float ymin0 = outputLocations[n * 4 + 1];
			float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
			float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

			float xmin1 = outputLocations[m * 4 + 0];
			float ymin1 = outputLocations[m * 4 + 1];
			float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
			float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

			// 计算两个矩形框的重叠度（IOU）
			float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

			// 如果重叠度大于阈值，则将后续检测框标记为无效
			if (iou > threshold) {
				order[j] = -1;
			}
		}
	}
	return 0; // 返回成功
}

/**
 * @brief 快速排序索引数组，逆序排序
 *
 * @param input 输入数组
 * @param left 左边界
 * @param right 右边界
 * @param indices 索引数组
 * @return int 分区点索引
 */
static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices) {
	float key;
	int   key_index;
	int   low = left;
	int   high = right;

	if (left < right) {
		key_index = indices[left];
		key = input[left];

		while (low < high) {
			// 从右往左找到第一个比 key 小的元素
			while (low < high && input[high] <= key) {
				high--;
			}
			// 将该元素移动到 low 的位置
			input[low] = input[high];
			indices[low] = indices[high];

			// 从左往右找到第一个比 key 大的元素
			while (low < high && input[low] >= key) {
				low++;
			}
			// 将该元素移动到 high 的位置
			input[high] = input[low];
			indices[high] = indices[low];
		}

		// 将 key 放到 low 的位置
		input[low] = key;
		indices[low] = key_index;

		// 递归对左右两部分进行排序
		quick_sort_indice_inverse(input, left, low - 1, indices);
		quick_sort_indice_inverse(input, low + 1, right, indices);
	}
	return low; // 返回分区点索引
}

/**
 * @brief Sigmoid 函数
 *
 * @param x 输入值
 * @return float 输出值
 */
static float sigmoid(float x) {
	return 1.0 / (1.0 + expf(-x)); // 计算 Sigmoid 函数值
}

/**
 * @brief 反 Sigmoid 函数
 *
 * @param y 输入值
 * @return float 输出值
 */
static float unsigmoid(float y) {
	return -1.0 * logf((1.0 / y) - 1.0); // 计算反 Sigmoid 函数值
}

/**
 * @brief 将值限制在指定范围内
 *
 * @param val 输入值
 * @param min 最小值
 * @param max 最大值
 * @return int32_t 限制后的值
 */
inline static int32_t __clip(float val, float min, float max) {
	float f = val <= min ? min : (val >= max ? max : val); // 将值限制在指定范围内
	return f; // 返回限制后的值
}

/**
 * @brief 将浮点数量化为固定点表示
 *
 * @param f32 浮点数
 * @param zp 零点值
 * @param scale 缩放因子
 * @return int8_t 量化后的值
 */
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
	float dst_val = (f32 / scale) + zp; // 计算量化后的值
	int8_t res = (int8_t)__clip(dst_val, -128, 127); // 将值限制在 [-128, 127] 范围内
	return res; // 返回量化后的值
}

/**
 * @brief 将量化后的值反量化为浮点数
 *
 * @param qnt 量化后的值
 * @param zp 零点值
 * @param scale 缩放因子
 * @return float 反量化后的浮点数
 */
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
	return ((float)qnt - (float)zp) * scale; // 计算反量化后的浮点数
}

/**
 * @brief 处理模型输出，解析目标框的位置、类别和置信度等信息
 *
 * @param input 模型输出数据指针
 * @param anchor 锚点数组
 * @param grid_h 网格高度
 * @param grid_w 网格宽度
 * @param height 输入图像高度
 * @param width 输入图像宽度
 * @param stride 步长
 * @param boxes 存储目标框的位置信息
 * @param objProbs 存储目标框的置信度
 * @param classId 存储目标框的类别
 * @param threshold 阈值
 * @param zp 零点值
 * @param scale 缩放因子
 * @return int 有效目标框的数量
 */
static int process(int8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
	std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
	int32_t zp, float scale) {
	int    validCount = 0; // 有效目标框的数量
	int    grid_len = grid_h * grid_w; // 网格大小
	float  thres = unsigmoid(threshold); // 将阈值反量化为浮点数
	int8_t thres_i8 = qnt_f32_to_affine(thres, zp, scale); // 将浮点数阈值量化为8位固定点表示

	for (int a = 0; a < 3; a++) { // 遍历三个不同尺度的预测框
		for (int i = 0; i < grid_h; i++) {
			for (int j = 0; j < grid_w; j++) {
				int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j]; // 目标框置信度
				if (box_confidence >= thres_i8) { // 如果目标框置信度超过阈值
					int     offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
					int8_t* in_ptr = input + offset;

					// 解析目标框的位置信息
					float   box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
					float   box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
					float   box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
					float   box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
					box_x = (box_x + j) * (float)stride;
					box_y = (box_y + i) * (float)stride;
					box_w = box_w * box_w * (float)anchor[a * 2];
					box_h = box_h * box_h * (float)anchor[a * 2 + 1];
					box_x -= (box_w / 2.0);
					box_y -= (box_h / 2.0);

					// 解析目标框的类别信息和置信度
					int8_t maxClassProbs = in_ptr[5 * grid_len];
					int    maxClassId = 0;
					for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
						int8_t prob = in_ptr[(5 + k) * grid_len];
						if (prob > maxClassProbs) {
							maxClassId = k;
							maxClassProbs = prob;
						}
					}
					if (maxClassProbs > thres_i8) { // 如果类别置信度超过阈值
						objProbs.push_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale)) * sigmoid(deqnt_affine_to_f32(box_confidence, zp, scale)));
						classId.push_back(maxClassId);
						validCount++;
						boxes.push_back(box_x);
						boxes.push_back(box_y);
						boxes.push_back(box_w);
						boxes.push_back(box_h);
					}
				}
			}
		}
	}
	return validCount; // 返回有效目标框的数量
}

/**
 * @brief 后处理函数，用于处理模型输出并生成检测结果
 *
 * @param input0 指向第一个输入张量的缓冲区
 * @param input1 指向第二个输入张量的缓冲区
 * @param input2 指向第三个输入张量的缓冲区
 * @param model_in_h 模型输入图像的高度
 * @param model_in_w 模型输入图像的宽度
 * @param conf_threshold 目标置信度阈值
 * @param nms_threshold NMS 阈值
 * @param scale_w 图像宽度缩放比例
 * @param scale_h 图像高度缩放比例
 * @param qnt_zps 输出 zero points 数组
 * @param qnt_scales 输出 scales 数组
 * @param group 检测结果组指针
 * @return int 返回值为 0 表示后处理成功
 */
int YoloV5Detector::post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales, detect_result_group_t* group)
{
	static int init = -1; // 初始化标志
	if (init == -1) {
		int ret = 0;
		ret = loadLabelName(LABEL_NALE_TXT_PATH, labels, OBJ_CLASS_NUM); // 加载标签名称
		if (ret < 0) {
			return -1;
		}

		init = 0; // 初始化完成
	}
	memset(group, 0, sizeof(detect_result_group_t)); // 清空检测结果组

	std::vector<float> filterBoxes; // 过滤后的目标框位置信息
	std::vector<float> objProbs;   // 目标框的置信度
	std::vector<int>   classId;    // 目标框的类别


	// stride 8
	int stride0 = 8;
	int grid_h0 = model_in_h / stride0;
	int grid_w0 = model_in_w / stride0;
	int validCount0 = 0;
	validCount0 = process(input0, (int*)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
		classId, conf_threshold, qnt_zps[0], qnt_scales[0]);  // 处理模型输出（stride=8）

	// stride 16
	int stride1 = 16;
	int grid_h1 = model_in_h / stride1;
	int grid_w1 = model_in_w / stride1;
	int validCount1 = 0;
	validCount1 = process(input1, (int*)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
		classId, conf_threshold, qnt_zps[1], qnt_scales[1]); // 处理模型输出（stride=16）

	// stride 32
	int stride2 = 32;
	int grid_h2 = model_in_h / stride2;
	int grid_w2 = model_in_w / stride2;
	int validCount2 = 0;
	validCount2 = process(input2, (int*)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
		classId, conf_threshold, qnt_zps[2], qnt_scales[2]); // 处理模型输出（stride=32）

	int validCount = validCount0 + validCount1 + validCount2;  // 计算有效目标框的总数
	// no object detect
	if (validCount <= 0) {
		return 0;
	}

	std::vector<int> indexArray;
	for (int i = 0; i < validCount; ++i) {
		indexArray.push_back(i);
	}

	quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray); // 对目标框的置信度进行排序

	std::set<int> class_set(std::begin(classId), std::end(classId)); // 获取目标框的类别集合

	for (auto c : class_set) {
		nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold); // 对每个类别进行非极大值抑制
	}
	int last_count = 0;
	group->count = 0;
	/* box valid detect target */
	for (int i = 0; i < validCount; ++i) {
		if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
			continue;
		}
		int n = indexArray[i];

		float x1 = filterBoxes[n * 4 + 0];
		float y1 = filterBoxes[n * 4 + 1];
		float x2 = x1 + filterBoxes[n * 4 + 2];
		float y2 = y1 + filterBoxes[n * 4 + 3];
		int   id = classId[n];
		float obj_conf = objProbs[i];

		group->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
		group->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
		group->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
		group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
		group->results[last_count].prop = obj_conf;
		char* label = labels[id];
		strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

		// printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
		// group->results[last_count].box.top,
		//        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
		last_count++;
	}
	group->count = last_count;

	return 0;
}


/**
 * @brief 执行推理过程
 *
 * @return int 返回值，0表示成功，-1表示失败
 */
int YoloV5Detector::run_inference(cv::Mat img)
{
	//cv::Mat img = original_image; // 获取原始图像
	original_image = img;
	// 获取图像宽高
	int src_width = original_image.cols;
	int src_height = original_image.rows;

	// 初始化rga上下文和缓冲区
	rga_buffer_t src;
	rga_buffer_t dst;
	memset(&src, 0, sizeof(src));
	memset(&dst, 0, sizeof(dst));
	im_rect src_rect;
	im_rect dst_rect;
	memset(&src_rect, 0, sizeof(src_rect));
	memset(&dst_rect, 0, sizeof(dst_rect));
	void* resize_buf = nullptr; // 缩放缓冲区

	// 将输入图像转换到模型需要的格式
	if (src_width != dst_width || src_height != dst_height)
	{
		resize_buf = malloc(dst_height * dst_width * channel_count); // 分配缩放缓冲区内存
		memset(resize_buf, 0x00, dst_height * dst_width * channel_count); // 初始化缩放缓冲区

		// 设置源和目标缓冲区
		src = wrapbuffer_virtualaddr((void*)img.data, src_width, src_height, RK_FORMAT_RGB_888);
		dst = wrapbuffer_virtualaddr((void*)resize_buf, dst_width, dst_height, RK_FORMAT_RGB_888);

		// 图像检查和缩放
		ret = imcheck(src, dst, src_rect, dst_rect);
		if (IM_STATUS_NOERROR != ret)
		{
			printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
			exit(-1);
		}
		IM_STATUS STATUS = imresize(src, dst);

		// 创建OpenCV Mat对象以便在后面使用
		cv::Mat resize_img(cv::Size(dst_width, dst_height), CV_8UC3, resize_buf);
		inputs[0].buf = resize_buf;
	}
	else
		inputs[0].buf = (void*)img.data; // 不需要缩放，直接使用原始图像数据

	// 设置rknn的输入数据
	rknn_inputs_set(rk_model, io_num.n_input, inputs);

	// 设置输出
	rknn_output outputs[io_num.n_output];
	memset(outputs, 0, sizeof(outputs));
	for (int i = 0; i < io_num.n_output; i++) {
		outputs[i].want_float = 0;
	}

	// 调用npu进行推演
	ret = rknn_run(rk_model, NULL);

	// 获取npu的推演输出结果
	ret = rknn_outputs_get(rk_model, io_num.n_output, outputs, NULL);

	// 进行后处理
	const float nms_threshold = NMS_THRESH; // 非极大值抑制的阈值
	const float box_conf_threshold = BOX_THRESH; // 目标框的置信度阈值
	float scale_w = (float)dst_width / src_width; // 宽度缩放比例
	float scale_h = (float)dst_height / src_height; // 高度缩放比例

	detect_result_group_t detect_result_group; // 检测结果组
	std::vector<float> out_scales;
	std::vector<int32_t> out_zps;
	for (int i = 0; i < io_num.n_output; ++i)
	{
		out_scales.push_back(output_attrs[i].scale);
		out_zps.push_back(output_attrs[i].zp);
	}

	// 进行后处理
	this->post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, dst_height, dst_width,
		box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

	// 绘制检测到的目标
	char text[256];
	for (int i = 0; i < detect_result_group.count; i++)
	{
		detect_result_t* det_result = &(detect_result_group.results[i]);
		sprintf(text, "%d: %s %.1f%%", i + 1, det_result->name, det_result->prop * 100);
		int x1 = det_result->box.left;
		int y1 = det_result->box.top;
		// 绘制目标框和标签
		rectangle(original_image, cv::Point(x1, y1), cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 0, 255, 0), 3);
		putText(original_image, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	}

	// 释放输出资源
	ret = rknn_outputs_release(rk_model, io_num.n_output, outputs);

	// 释放缩放缓冲区内存
	if (resize_buf)
	{
		free(resize_buf);
	}

	return 0; // 返回成功
}


/**
 * @brief RKNNYoloV5 类的构造函数，用于初始化 YOLOv5 模型
 *
 * @param model_name 模型文件名
 * @param n NPU 核心编号
 */
YoloV5Detector::YoloV5Detector(const char* model_name, int core_id)
{
	printf("Loading model...\n"); // 打印加载模型的信息

	int model_data_size = 0; // 模型数据大小

	// 读取模型文件数据
	model_data = load_model(model_name, &model_data_size);


	// 通过模型文件初始化 RKNN 类
	ret = rknn_init(&rk_model, model_data, model_data_size, 0, NULL);
	if (ret < 0) {
		printf("rknn_init error ret=%d\n", ret);
		exit(-1);
	}

	// 初始化rknn类的版本
	ret = rknn_query(rk_model, RKNN_QUERY_SDK_VERSION, &sdk_version, sizeof(rknn_sdk_version));

	if (ret < 0)
	{
		printf("rknn_init error ret=%d\n", ret);
		exit(-1);
	}
	else {
		printf("sdk api version: %s\n", sdk_version.api_version);
		printf("driver version: %s\n", sdk_version.drv_version);
	}

	// 设置 NPU 核心,RK3588 有三个核心
	rknn_core_mask core_mask;
	if (core_id == 0) {
		core_mask = RKNN_NPU_CORE_0;
	}
	else if (core_id == 1) {
		core_mask = RKNN_NPU_CORE_1;
	}
	else {
		core_mask = RKNN_NPU_CORE_2;
	}

	// 设置 NPU 核心掩码
	int ret = rknn_set_core_mask(rk_model, core_mask);
	if (ret < 0) {
		printf("rknn_set_core_mask error ret=%d\n", ret);
		exit(-1);
	}


	// 获取模型的输入参数
	ret = rknn_query(rk_model, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret < 0)
	{
		printf("rknn_init error ret=%d\n", ret);
		exit(-1);
	}

	// 设置输入数组
	input_attrs = new rknn_tensor_attr[io_num.n_input];
	memset(input_attrs, 0, sizeof(input_attrs));
	for (int i = 0; i < io_num.n_input; i++)
	{
		input_attrs[i].index = i;
		ret = rknn_query(rk_model, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
		if (ret < 0)
		{
			printf("rknn_init error ret=%d\n", ret);
			exit(-1);
		}
	}

	// 设置输出数组
	output_attrs = new rknn_tensor_attr[io_num.n_output];
	memset(output_attrs, 0, sizeof(output_attrs));
	for (int i = 0; i < io_num.n_output; i++)
	{
		output_attrs[i].index = i;
		ret = rknn_query(rk_model, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
	}

	// 设置输入参数
	if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
	{
		printf("model is NCHW input fmt\n");
		channel_count = input_attrs[0].dims[1];
		dst_height = input_attrs[0].dims[2];
		dst_width = input_attrs[0].dims[3];
	}
	else
	{
		printf("model is NHWC input fmt\n");
		dst_height = input_attrs[0].dims[1];
		dst_width = input_attrs[0].dims[2];
		channel_count = input_attrs[0].dims[3];
	}

	memset(inputs, 0, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].size = dst_width * dst_height * channel_count;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
	inputs[0].pass_through = 0;
}





/**
 * @brief 析构函数，用于释放资源
 */
YoloV5Detector::~YoloV5Detector()
{
	// 销毁模型
	ret = rknn_destroy(rk_model);

	// 释放输入和输出属性数组
	delete[] input_attrs;
	delete[] output_attrs;

	// 如果模型数据存在，则释放内存
	if (model_data) {
		free(model_data);
	}
}
