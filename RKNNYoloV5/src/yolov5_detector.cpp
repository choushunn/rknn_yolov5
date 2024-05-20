#include "yolov5_detector.h"

int YoloV5Detector::pre_process()
{
	return 0;
}


/**
 * @brief �����������ο���ص���
 *
 * @param xmin0 ��һ�����ο�����Ͻ� x ����
 * @param ymin0 ��һ�����ο�����Ͻ� y ����
 * @param xmax0 ��һ�����ο�����½� x ����
 * @param ymax0 ��һ�����ο�����½� y ����
 * @param xmin1 �ڶ������ο�����Ͻ� x ����
 * @param ymin1 �ڶ������ο�����Ͻ� y ����
 * @param xmax1 �ڶ������ο�����½� x ����
 * @param ymax1 �ڶ������ο�����½� y ����
 * @return float �ص��ȣ�IOU��
 */
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
	float xmin1, float ymin1, float xmax1, float ymax1) {
	float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0); // �������
	float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0); // �����߶�
	float i = w * h; // �������
	float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i; // �������
	return u <= 0.f ? 0.f : (i / u); // ���� IOU ������
}

/**
 * @brief NMS �Ǽ���ֵ����
 *
 * @param validCount ��Ч��������
 * @param outputLocations ����λ����Ϣ����
 * @param classIds �����������
 * @param order ������������
 * @param filterId ��Ҫ���˵���� ID
 * @param threshold IOU ��ֵ
 * @return int ����ֵΪ 0 ��ʾ�ɹ�
 */
static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds,
	std::vector<int>& order, int filterId, float threshold) {
	// �������м���
	for (int i = 0; i < validCount; ++i) {
		// �����ǰ������Ч�����ƥ�䣬������
		if (order[i] == -1 || classIds[i] != filterId) {
			continue;
		}
		int n = order[i];
		// ����������������
		for (int j = i + 1; j < validCount; ++j) {
			int m = order[j];
			// �����ǰ������Ч�����ƥ�䣬������
			if (m == -1 || classIds[i] != filterId) {
				continue;
			}
			// ��ȡ��ǰ�ͺ��������������Ϣ
			float xmin0 = outputLocations[n * 4 + 0];
			float ymin0 = outputLocations[n * 4 + 1];
			float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
			float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

			float xmin1 = outputLocations[m * 4 + 0];
			float ymin1 = outputLocations[m * 4 + 1];
			float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
			float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

			// �����������ο���ص��ȣ�IOU��
			float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

			// ����ص��ȴ�����ֵ���򽫺���������Ϊ��Ч
			if (iou > threshold) {
				order[j] = -1;
			}
		}
	}
	return 0; // ���سɹ�
}

/**
 * @brief ���������������飬��������
 *
 * @param input ��������
 * @param left ��߽�
 * @param right �ұ߽�
 * @param indices ��������
 * @return int ����������
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
			// ���������ҵ���һ���� key С��Ԫ��
			while (low < high && input[high] <= key) {
				high--;
			}
			// ����Ԫ���ƶ��� low ��λ��
			input[low] = input[high];
			indices[low] = indices[high];

			// ���������ҵ���һ���� key ���Ԫ��
			while (low < high && input[low] >= key) {
				low++;
			}
			// ����Ԫ���ƶ��� high ��λ��
			input[high] = input[low];
			indices[high] = indices[low];
		}

		// �� key �ŵ� low ��λ��
		input[low] = key;
		indices[low] = key_index;

		// �ݹ�����������ֽ�������
		quick_sort_indice_inverse(input, left, low - 1, indices);
		quick_sort_indice_inverse(input, low + 1, right, indices);
	}
	return low; // ���ط���������
}

/**
 * @brief Sigmoid ����
 *
 * @param x ����ֵ
 * @return float ���ֵ
 */
static float sigmoid(float x) {
	return 1.0 / (1.0 + expf(-x)); // ���� Sigmoid ����ֵ
}

/**
 * @brief �� Sigmoid ����
 *
 * @param y ����ֵ
 * @return float ���ֵ
 */
static float unsigmoid(float y) {
	return -1.0 * logf((1.0 / y) - 1.0); // ���㷴 Sigmoid ����ֵ
}

/**
 * @brief ��ֵ������ָ����Χ��
 *
 * @param val ����ֵ
 * @param min ��Сֵ
 * @param max ���ֵ
 * @return int32_t ���ƺ��ֵ
 */
inline static int32_t __clip(float val, float min, float max) {
	float f = val <= min ? min : (val >= max ? max : val); // ��ֵ������ָ����Χ��
	return f; // �������ƺ��ֵ
}

/**
 * @brief ������������Ϊ�̶����ʾ
 *
 * @param f32 ������
 * @param zp ���ֵ
 * @param scale ��������
 * @return int8_t �������ֵ
 */
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
	float dst_val = (f32 / scale) + zp; // �����������ֵ
	int8_t res = (int8_t)__clip(dst_val, -128, 127); // ��ֵ������ [-128, 127] ��Χ��
	return res; // �����������ֵ
}

/**
 * @brief ���������ֵ������Ϊ������
 *
 * @param qnt �������ֵ
 * @param zp ���ֵ
 * @param scale ��������
 * @return float ��������ĸ�����
 */
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
	return ((float)qnt - (float)zp) * scale; // ���㷴������ĸ�����
}

/**
 * @brief ����ģ�����������Ŀ����λ�á��������Ŷȵ���Ϣ
 *
 * @param input ģ���������ָ��
 * @param anchor ê������
 * @param grid_h ����߶�
 * @param grid_w ������
 * @param height ����ͼ��߶�
 * @param width ����ͼ����
 * @param stride ����
 * @param boxes �洢Ŀ����λ����Ϣ
 * @param objProbs �洢Ŀ�������Ŷ�
 * @param classId �洢Ŀ�������
 * @param threshold ��ֵ
 * @param zp ���ֵ
 * @param scale ��������
 * @return int ��ЧĿ��������
 */
static int process(int8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
	std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
	int32_t zp, float scale) {
	int    validCount = 0; // ��ЧĿ��������
	int    grid_len = grid_h * grid_w; // �����С
	float  thres = unsigmoid(threshold); // ����ֵ������Ϊ������
	int8_t thres_i8 = qnt_f32_to_affine(thres, zp, scale); // ����������ֵ����Ϊ8λ�̶����ʾ

	for (int a = 0; a < 3; a++) { // ����������ͬ�߶ȵ�Ԥ���
		for (int i = 0; i < grid_h; i++) {
			for (int j = 0; j < grid_w; j++) {
				int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j]; // Ŀ������Ŷ�
				if (box_confidence >= thres_i8) { // ���Ŀ������Ŷȳ�����ֵ
					int     offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
					int8_t* in_ptr = input + offset;

					// ����Ŀ����λ����Ϣ
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

					// ����Ŀ���������Ϣ�����Ŷ�
					int8_t maxClassProbs = in_ptr[5 * grid_len];
					int    maxClassId = 0;
					for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
						int8_t prob = in_ptr[(5 + k) * grid_len];
						if (prob > maxClassProbs) {
							maxClassId = k;
							maxClassProbs = prob;
						}
					}
					if (maxClassProbs > thres_i8) { // ���������Ŷȳ�����ֵ
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
	return validCount; // ������ЧĿ��������
}

/**
 * @brief �����������ڴ���ģ����������ɼ����
 *
 * @param input0 ָ���һ�����������Ļ�����
 * @param input1 ָ��ڶ������������Ļ�����
 * @param input2 ָ����������������Ļ�����
 * @param model_in_h ģ������ͼ��ĸ߶�
 * @param model_in_w ģ������ͼ��Ŀ��
 * @param conf_threshold Ŀ�����Ŷ���ֵ
 * @param nms_threshold NMS ��ֵ
 * @param scale_w ͼ�������ű���
 * @param scale_h ͼ��߶����ű���
 * @param qnt_zps ��� zero points ����
 * @param qnt_scales ��� scales ����
 * @param group �������ָ��
 * @return int ����ֵΪ 0 ��ʾ����ɹ�
 */
int YoloV5Detector::post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales, detect_result_group_t* group)
{
	static int init = -1; // ��ʼ����־
	if (init == -1) {
		int ret = 0;
		ret = loadLabelName(LABEL_NALE_TXT_PATH, labels, OBJ_CLASS_NUM); // ���ر�ǩ����
		if (ret < 0) {
			return -1;
		}

		init = 0; // ��ʼ�����
	}
	memset(group, 0, sizeof(detect_result_group_t)); // ��ռ������

	std::vector<float> filterBoxes; // ���˺��Ŀ���λ����Ϣ
	std::vector<float> objProbs;   // Ŀ�������Ŷ�
	std::vector<int>   classId;    // Ŀ�������


	// stride 8
	int stride0 = 8;
	int grid_h0 = model_in_h / stride0;
	int grid_w0 = model_in_w / stride0;
	int validCount0 = 0;
	validCount0 = process(input0, (int*)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
		classId, conf_threshold, qnt_zps[0], qnt_scales[0]);  // ����ģ�������stride=8��

	// stride 16
	int stride1 = 16;
	int grid_h1 = model_in_h / stride1;
	int grid_w1 = model_in_w / stride1;
	int validCount1 = 0;
	validCount1 = process(input1, (int*)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
		classId, conf_threshold, qnt_zps[1], qnt_scales[1]); // ����ģ�������stride=16��

	// stride 32
	int stride2 = 32;
	int grid_h2 = model_in_h / stride2;
	int grid_w2 = model_in_w / stride2;
	int validCount2 = 0;
	validCount2 = process(input2, (int*)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
		classId, conf_threshold, qnt_zps[2], qnt_scales[2]); // ����ģ�������stride=32��

	int validCount = validCount0 + validCount1 + validCount2;  // ������ЧĿ��������
	// no object detect
	if (validCount <= 0) {
		return 0;
	}

	std::vector<int> indexArray;
	for (int i = 0; i < validCount; ++i) {
		indexArray.push_back(i);
	}

	quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray); // ��Ŀ�������ŶȽ�������

	std::set<int> class_set(std::begin(classId), std::end(classId)); // ��ȡĿ������𼯺�

	for (auto c : class_set) {
		nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold); // ��ÿ�������зǼ���ֵ����
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
 * @brief ִ���������
 *
 * @return int ����ֵ��0��ʾ�ɹ���-1��ʾʧ��
 */
int YoloV5Detector::run_inference(cv::Mat img)
{
	//cv::Mat img = original_image; // ��ȡԭʼͼ��
	original_image = img;
	// ��ȡͼ����
	int src_width = original_image.cols;
	int src_height = original_image.rows;

	// ��ʼ��rga�����ĺͻ�����
	rga_buffer_t src;
	rga_buffer_t dst;
	memset(&src, 0, sizeof(src));
	memset(&dst, 0, sizeof(dst));
	im_rect src_rect;
	im_rect dst_rect;
	memset(&src_rect, 0, sizeof(src_rect));
	memset(&dst_rect, 0, sizeof(dst_rect));
	void* resize_buf = nullptr; // ���Ż�����

	// ������ͼ��ת����ģ����Ҫ�ĸ�ʽ
	if (src_width != dst_width || src_height != dst_height)
	{
		resize_buf = malloc(dst_height * dst_width * channel_count); // �������Ż������ڴ�
		memset(resize_buf, 0x00, dst_height * dst_width * channel_count); // ��ʼ�����Ż�����

		// ����Դ��Ŀ�껺����
		src = wrapbuffer_virtualaddr((void*)img.data, src_width, src_height, RK_FORMAT_RGB_888);
		dst = wrapbuffer_virtualaddr((void*)resize_buf, dst_width, dst_height, RK_FORMAT_RGB_888);

		// ͼ���������
		ret = imcheck(src, dst, src_rect, dst_rect);
		if (IM_STATUS_NOERROR != ret)
		{
			printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
			exit(-1);
		}
		IM_STATUS STATUS = imresize(src, dst);

		// ����OpenCV Mat�����Ա��ں���ʹ��
		cv::Mat resize_img(cv::Size(dst_width, dst_height), CV_8UC3, resize_buf);
		inputs[0].buf = resize_buf;
	}
	else
		inputs[0].buf = (void*)img.data; // ����Ҫ���ţ�ֱ��ʹ��ԭʼͼ������

	// ����rknn����������
	rknn_inputs_set(rk_model, io_num.n_input, inputs);

	// �������
	rknn_output outputs[io_num.n_output];
	memset(outputs, 0, sizeof(outputs));
	for (int i = 0; i < io_num.n_output; i++) {
		outputs[i].want_float = 0;
	}

	// ����npu��������
	ret = rknn_run(rk_model, NULL);

	// ��ȡnpu������������
	ret = rknn_outputs_get(rk_model, io_num.n_output, outputs, NULL);

	// ���к���
	const float nms_threshold = NMS_THRESH; // �Ǽ���ֵ���Ƶ���ֵ
	const float box_conf_threshold = BOX_THRESH; // Ŀ�������Ŷ���ֵ
	float scale_w = (float)dst_width / src_width; // ������ű���
	float scale_h = (float)dst_height / src_height; // �߶����ű���

	detect_result_group_t detect_result_group; // �������
	std::vector<float> out_scales;
	std::vector<int32_t> out_zps;
	for (int i = 0; i < io_num.n_output; ++i)
	{
		out_scales.push_back(output_attrs[i].scale);
		out_zps.push_back(output_attrs[i].zp);
	}

	// ���к���
	this->post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, dst_height, dst_width,
		box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

	// ���Ƽ�⵽��Ŀ��
	char text[256];
	for (int i = 0; i < detect_result_group.count; i++)
	{
		detect_result_t* det_result = &(detect_result_group.results[i]);
		sprintf(text, "%d: %s %.1f%%", i + 1, det_result->name, det_result->prop * 100);
		int x1 = det_result->box.left;
		int y1 = det_result->box.top;
		// ����Ŀ���ͱ�ǩ
		rectangle(original_image, cv::Point(x1, y1), cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 0, 255, 0), 3);
		putText(original_image, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	}

	// �ͷ������Դ
	ret = rknn_outputs_release(rk_model, io_num.n_output, outputs);

	// �ͷ����Ż������ڴ�
	if (resize_buf)
	{
		free(resize_buf);
	}

	return 0; // ���سɹ�
}


/**
 * @brief RKNNYoloV5 ��Ĺ��캯�������ڳ�ʼ�� YOLOv5 ģ��
 *
 * @param model_name ģ���ļ���
 * @param n NPU ���ı��
 */
YoloV5Detector::YoloV5Detector(const char* model_name, int core_id)
{
	printf("Loading model...\n"); // ��ӡ����ģ�͵���Ϣ

	int model_data_size = 0; // ģ�����ݴ�С

	// ��ȡģ���ļ�����
	model_data = load_model(model_name, &model_data_size);


	// ͨ��ģ���ļ���ʼ�� RKNN ��
	ret = rknn_init(&rk_model, model_data, model_data_size, 0, NULL);
	if (ret < 0) {
		printf("rknn_init error ret=%d\n", ret);
		exit(-1);
	}

	// ��ʼ��rknn��İ汾
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

	// ���� NPU ����,RK3588 ����������
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

	// ���� NPU ��������
	int ret = rknn_set_core_mask(rk_model, core_mask);
	if (ret < 0) {
		printf("rknn_set_core_mask error ret=%d\n", ret);
		exit(-1);
	}


	// ��ȡģ�͵��������
	ret = rknn_query(rk_model, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
	if (ret < 0)
	{
		printf("rknn_init error ret=%d\n", ret);
		exit(-1);
	}

	// ������������
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

	// �����������
	output_attrs = new rknn_tensor_attr[io_num.n_output];
	memset(output_attrs, 0, sizeof(output_attrs));
	for (int i = 0; i < io_num.n_output; i++)
	{
		output_attrs[i].index = i;
		ret = rknn_query(rk_model, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
	}

	// �����������
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
 * @brief ���������������ͷ���Դ
 */
YoloV5Detector::~YoloV5Detector()
{
	// ����ģ��
	ret = rknn_destroy(rk_model);

	// �ͷ�����������������
	delete[] input_attrs;
	delete[] output_attrs;

	// ���ģ�����ݴ��ڣ����ͷ��ڴ�
	if (model_data) {
		free(model_data);
	}
}
