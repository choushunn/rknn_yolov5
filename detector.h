#ifndef _RKNN_DEMO_H_
#define _RKNN_DEMO_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <vector>
#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"
#include "file_utils.h"

// 定义目标名称的最大尺寸
#define OBJ_NAME_MAX_SIZE 64

// 定义目标数量的最大尺寸 
#define OBJ_NUMB_MAX_SIZE 128

// 定义目标类别的数量
#define OBJ_CLASS_NUM 4

// 定义非极大值抑制(NMS)的阈值
#define NMS_THRESH 0.45

// 定义边框置信度的阈值
#define BOX_THRESH 0.45

// 定义属性框的大小
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

// 目标检测操作的结果
typedef struct {
	// 检测到的目标的边界框
	image_rect_t box;
	// 检测的置信度得分
	float prop;
	// 检测到的目标的类别ID
	int cls_id;
} object_detect_result;

// 目标检测结果列表
typedef struct {
	// 检测结果列表的ID
	int id;
	// 检测到的目标数量
	int count;
	// 目标检测结果数组
	object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

// RKNN 上下文
typedef struct {
	// RKNN上下文
	rknn_context rknn_ctx;
	// 输入和输出张量的数量
	rknn_input_output_num io_num;
	// 输入张量的属性
	rknn_tensor_attr* input_attrs;
	// 输出张量的属性
	rknn_tensor_attr* output_attrs;
	// 模型的通道数
	int model_channel;
	// 模型的宽度
	int model_width;
	// 模型的高度
	int model_height;
	// 标识模型是否量化
	bool is_quant;
} rknn_app_context_t;


// 存储类别标签的数组
static char* labels[OBJ_CLASS_NUM] = {
	"1",
	"2",
	"3",
	"4"
};

// 锚框的尺寸和比例
const int anchor[3][6] = {
	{10, 13, 16, 30, 33, 23},
	{30, 61, 62, 45, 59, 119},
	{116, 90, 156, 198, 373, 326} 
};

// 将一个值限制在指定的范围内
inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

// 从文件中读取一行数据
static char* readLine(FILE* fp, char* buffer, int* len)
{
	int ch;
	int i = 0;
	size_t buff_len = 0;

	// 动态分配缓冲区内存
	buffer = (char*)malloc(buff_len + 1);
	if (!buffer)
		return NULL; // 内存分配失败

	// 循环读取一行数据
	while ((ch = fgetc(fp)) != '\n' && ch != EOF)
	{
		buff_len++;
		void* tmp = realloc(buffer, buff_len + 1);
		if (tmp == NULL)
		{
			free(buffer);
			return NULL; // 内存分配失败
		}
		buffer = (char*)tmp;

		buffer[i] = (char)ch;
		i++;
	}
	buffer[i] = '\0';

	*len = buff_len;

	// 检测读取结束条件
	if (ch == EOF && (i == 0 || ferror(fp)))
	{
		free(buffer);
		return NULL;
	}
	return buffer;
}

// 读取标签文件中的所有行
static int readLines(const char* fileName, char* lines[], int max_line)
{
	FILE* file = fopen(fileName, "r");
	char* s;
	int i = 0;
	int n = 0;

	if (file == NULL)
	{
		printf("Open %s fail!\n", fileName);
		return -1;
	}

	// 读取每一行并存储到数组中
	while ((s = readLine(file, s, &n)) != NULL)
	{
		lines[i++] = s;
		if (i >= max_line)
			break;
	}
	fclose(file);
	return i;
}

// 加载标签名称
static int loadLabelName(const char* locationFilename, char* label[])
{
	printf("load lable %s\n", locationFilename);
	readLines(locationFilename, label, OBJ_CLASS_NUM);
	return 0;
}


static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
	float ymax1)
{
	float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
	float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
	float i = w * h;
	float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
	return u <= 0.f ? 0.f : (i / u);
}
// 非极大值抑制
static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
	int filterId, float threshold)
{
	for (int i = 0; i < validCount; ++i)
	{
		if (order[i] == -1 || classIds[i] != filterId)
		{
			continue;
		}
		int n = order[i];
		for (int j = i + 1; j < validCount; ++j)
		{
			int m = order[j];
			if (m == -1 || classIds[i] != filterId)
			{
				continue;
			}
			float xmin0 = outputLocations[n * 4 + 0];
			float ymin0 = outputLocations[n * 4 + 1];
			float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
			float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

			float xmin1 = outputLocations[m * 4 + 0];
			float ymin1 = outputLocations[m * 4 + 1];
			float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
			float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

			float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

			if (iou > threshold)
			{
				order[j] = -1;
			}
		}
	}
	return 0;
}
/**
 * @brief 快速排序并记录索引的逆序函数
 * @param input 要排序的float数组
 * @param left 排序区间的左边界
 * @param right 排序区间的右边界
 * @param indices 与input对应的索引数组
 * @return 排序后中间值的索引
 */
static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
	float key;
	int key_index;
	int low = left;
	int high = right;
	if (left < right)
	{
		key_index = indices[left];
		key = input[left];
		while (low < high)
		{
			while (low < high && input[high] <= key)
			{
				high--;
			}
			input[low] = input[high];
			indices[low] = indices[high];
			while (low < high && input[low] >= key)
			{
				low++;
			}
			input[high] = input[low];
			indices[high] = indices[low];
		}
		input[low] = key;
		indices[low] = key_index;
		quick_sort_indice_inverse(input, left, low - 1, indices);
		quick_sort_indice_inverse(input, low + 1, right, indices);
	}
	return low;
}

/**
 * @brief Sigmoid函数
 * @param x 输入值
 * @return Sigmoid函数的输出值
 */
static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

/**
 * @brief Sigmoid函数的逆函数
 * @param y 输入值
 * @return Sigmoid函数逆函数的输出值
 */
static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

/**
 * @brief 限幅函数
 * @param val 输入值
 * @param min 最小值
 * @param max 最大值
 * @return 限幅后的值
 */
inline static int32_t __clip(float val, float min, float max)
{
	float f = val <= min ? min : (val >= max ? max : val);
	return f;
}

/**
 * @brief 将float32值量化为affine量化的int8_t值
 * @param f32 要量化的float32值
 * @param zp affine 量化的零点
 * @param scale affine 量化的缩放因子
 * @return 量化后的int8_t值
 */
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
	float dst_val = (f32 / scale) + zp;
	int8_t res = (int8_t)__clip(dst_val, -128, 127);
	return res;
}

/**
 * @brief 将float32值量化为affine量化的uint8_t值
 * @param f32 要量化的float32值
 * @param zp affine量化的零点
 * @param scale affine量化的缩放因子
 * @return 量化后的uint8_t值
 */
static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale)
{
	float dst_val = (f32 / scale) + zp;
	uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
	return res;
}

/**
 * @brief 将affine量化的int8_t值反量化为float32值
 * @param qnt 要反量化的int8_t值
 * @param zp affine量化的零点
 * @param scale affine量化的缩放因子
 * @return 反量化后的float32值
 */
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

/**
 * @brief 将affine量化的uint8_t值反量化为float32值
 * @param qnt 要反量化的uint8_t值
 * @param zp affine量化的零点
 * @param scale affine量化的缩放因子
 * @return 反量化后的float32值
 */
static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static int process_u8(uint8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
	std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
	int32_t zp, float scale)
{
	int validCount = 0;
	int grid_len = grid_h * grid_w;
	uint8_t thres_u8 = qnt_f32_to_affine_u8(threshold, zp, scale);
	for (int a = 0; a < 3; a++)
	{
		for (int i = 0; i < grid_h; i++)
		{
			for (int j = 0; j < grid_w; j++)
			{
				uint8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
				if (box_confidence >= thres_u8)
				{
					int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
					uint8_t* in_ptr = input + offset;
					float box_x = (deqnt_affine_u8_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
					float box_y = (deqnt_affine_u8_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
					float box_w = (deqnt_affine_u8_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
					float box_h = (deqnt_affine_u8_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
					box_x = (box_x + j) * (float)stride;
					box_y = (box_y + i) * (float)stride;
					box_w = box_w * box_w * (float)anchor[a * 2];
					box_h = box_h * box_h * (float)anchor[a * 2 + 1];
					box_x -= (box_w / 2.0);
					box_y -= (box_h / 2.0);

					uint8_t maxClassProbs = in_ptr[5 * grid_len];
					int maxClassId = 0;
					for (int k = 1; k < OBJ_CLASS_NUM; ++k)
					{
						uint8_t prob = in_ptr[(5 + k) * grid_len];
						if (prob > maxClassProbs)
						{
							maxClassId = k;
							maxClassProbs = prob;
						}
					}
					if (maxClassProbs > thres_u8)
					{
						objProbs.push_back((deqnt_affine_u8_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_u8_to_f32(box_confidence, zp, scale)));
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
	return validCount;
}

static int process_i8(int8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
	std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
	int32_t zp, float scale)
{
	int validCount = 0;
	int grid_len = grid_h * grid_w;
	int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
	for (int a = 0; a < 3; a++)
	{
		for (int i = 0; i < grid_h; i++)
		{
			for (int j = 0; j < grid_w; j++)
			{
				int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
				if (box_confidence >= thres_i8)
				{
					int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
					int8_t* in_ptr = input + offset;
					float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
					float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
					float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
					float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
					box_x = (box_x + j) * (float)stride;
					box_y = (box_y + i) * (float)stride;
					box_w = box_w * box_w * (float)anchor[a * 2];
					box_h = box_h * box_h * (float)anchor[a * 2 + 1];
					box_x -= (box_w / 2.0);
					box_y -= (box_h / 2.0);

					int8_t maxClassProbs = in_ptr[5 * grid_len];
					int maxClassId = 0;
					for (int k = 1; k < OBJ_CLASS_NUM; ++k)
					{
						int8_t prob = in_ptr[(5 + k) * grid_len];
						if (prob > maxClassProbs)
						{
							maxClassId = k;
							maxClassProbs = prob;
						}
					}
					if (maxClassProbs > thres_i8)
					{
						objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
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
	return validCount;
}

static int process_fp32(float* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
	std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold)
{
	int validCount = 0;
	int grid_len = grid_h * grid_w;

	for (int a = 0; a < 3; a++)
	{
		for (int i = 0; i < grid_h; i++)
		{
			for (int j = 0; j < grid_w; j++)
			{
				float box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
				if (box_confidence >= threshold)
				{
					int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
					float* in_ptr = input + offset;
					float box_x = *in_ptr * 2.0 - 0.5;
					float box_y = in_ptr[grid_len] * 2.0 - 0.5;
					float box_w = in_ptr[2 * grid_len] * 2.0;
					float box_h = in_ptr[3 * grid_len] * 2.0;
					box_x = (box_x + j) * (float)stride;
					box_y = (box_y + i) * (float)stride;
					box_w = box_w * box_w * (float)anchor[a * 2];
					box_h = box_h * box_h * (float)anchor[a * 2 + 1];
					box_x -= (box_w / 2.0);
					box_y -= (box_h / 2.0);

					float maxClassProbs = in_ptr[5 * grid_len];
					int maxClassId = 0;
					for (int k = 1; k < OBJ_CLASS_NUM; ++k)
					{
						float prob = in_ptr[(5 + k) * grid_len];
						if (prob > maxClassProbs)
						{
							maxClassId = k;
							maxClassProbs = prob;
						}
					}
					if (maxClassProbs > threshold)
					{
						objProbs.push_back(maxClassProbs * box_confidence);
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
	return validCount;
}

/**
 * @brief 打印rknn_tensor_attr结构体的内容
 * @param attr rknn_tensor_attr结构体指针
 */
static void dump_tensor_attr(rknn_tensor_attr* attr)
{
	printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
		"zp=%d, scale=%f\n",
		attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
		attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
		get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

class Detector {
private:
	rknn_app_context_t* app_ctx;
public:
	Detector(const char* model_path) {
		// 如果 model_path 为空,则使用默认路径
		if (model_path == nullptr || strlen(model_path) == 0)
		{
			// 将默认路径设置为 "/path/to/default/model.rknn"
			model_path = "./model/best.rknn";
		}
		app_ctx = new rknn_app_context_t;
		memset(app_ctx, 0, sizeof(rknn_app_context_t));
		int ret;
		int model_len = 0;
		char* model;
		rknn_context ctx = 0;

		// Load RKNN Model
		model_len = read_data_from_file(model_path, &model);
		if (model == NULL)
		{
			printf("rknn 模型加载失败!\n");
		}
		ret = rknn_init(&ctx, model, model_len, 0, NULL);
		free(model);
		if (ret < 0)
		{
			printf("rknn 初始化失败! ret=%d\n", ret);
		}
		// 模型初始之前先查询版本
		rknn_sdk_version sdk_version;   // RKNN SDK 版本号
		ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_version, sizeof(rknn_sdk_version));

		if (ret < 0)
		{
			printf("rknn_init error ret=%d\n", ret);
			exit(-1);
		}
		else {
			printf("rknn sdk api version: %s\n", sdk_version.api_version);
			printf("rknn driver version: %s\n", sdk_version.drv_version);
		}
		// Get Model Input Output Number
		rknn_input_output_num io_num;
		ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
		if (ret != RKNN_SUCC)
		{
			printf("rknn_query fail! ret=%d\n", ret);;
		}
		printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

		// Get Model Input Info
		printf("input tensors:\n");
		rknn_tensor_attr input_attrs[io_num.n_input];
		memset(input_attrs, 0, sizeof(input_attrs));
		for (int i = 0; i < io_num.n_input; i++)
		{
			input_attrs[i].index = i;
			ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
			if (ret != RKNN_SUCC)
			{
				printf("rknn_query fail! ret=%d\n", ret);
			}
			dump_tensor_attr(&(input_attrs[i]));
		}

		// Get Model Output Info
		printf("output tensors:\n");
		rknn_tensor_attr output_attrs[io_num.n_output];
		memset(output_attrs, 0, sizeof(output_attrs));
		for (int i = 0; i < io_num.n_output; i++)
		{
			output_attrs[i].index = i;
			ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
			if (ret != RKNN_SUCC)
			{
				printf("rknn_query fail! ret=%d\n", ret);
				//return -1;
			}
			dump_tensor_attr(&(output_attrs[i]));
		}
		// Set to context
		app_ctx->rknn_ctx = ctx;
		// TODO
		if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
		{
			app_ctx->is_quant = true;
			printf("rknn is quant.\n");
		}
		else
		{
			app_ctx->is_quant = false;
		}
		app_ctx->io_num = io_num;
		app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
		memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
		app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
		memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

		if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
		{
			printf("model is NCHW input fmt\n");
			app_ctx->model_channel = input_attrs[0].dims[1];
			app_ctx->model_height = input_attrs[0].dims[2];
			app_ctx->model_width = input_attrs[0].dims[3];
		}
		else
		{
			printf("model is NHWC input fmt\n");
			app_ctx->model_height = input_attrs[0].dims[1];
			app_ctx->model_width = input_attrs[0].dims[2];
			app_ctx->model_channel = input_attrs[0].dims[3];
		}
		printf("model input height=%d, width=%d, channel=%d\n",
			app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);
	}
	// 析构函数
	~Detector() {
		if (app_ctx->input_attrs != NULL)
		{
			free(app_ctx->input_attrs);
			app_ctx->input_attrs = NULL;
		}
		if (app_ctx->output_attrs != NULL)
		{
			free(app_ctx->output_attrs);
			app_ctx->output_attrs = NULL;
		}
		if (app_ctx->rknn_ctx != 0)
		{
			rknn_destroy(app_ctx->rknn_ctx);
			app_ctx->rknn_ctx = 0;
		}
	}

	int run(image_buffer_t* img, object_detect_result_list* od_results) {
		int ret;
		image_buffer_t dst_img;
		letterbox_t letter_box;
		rknn_input inputs[app_ctx->io_num.n_input];
		rknn_output outputs[app_ctx->io_num.n_output];
		const float nms_threshold = NMS_THRESH;      // Default NMS threshold
		const float box_conf_threshold = BOX_THRESH; // Default box threshold
		int bg_color = 114;

		if ((!app_ctx) || !(img) || (!od_results))
		{
			return -1;
		}

		memset(od_results, 0x00, sizeof(*od_results));
		memset(&letter_box, 0, sizeof(letterbox_t));
		memset(&dst_img, 0, sizeof(image_buffer_t));
		memset(inputs, 0, sizeof(inputs));
		memset(outputs, 0, sizeof(outputs));

		// Pre Process
		dst_img.width = app_ctx->model_width;
		dst_img.height = app_ctx->model_height;
		dst_img.format = IMAGE_FORMAT_RGB888;
		dst_img.size = get_image_size(&dst_img);
		dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);
		if (dst_img.virt_addr == NULL)
		{
			printf("malloc buffer size:%d fail!\n", dst_img.size);
			return -1;
		}
		// letterbox
		ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
		if (ret < 0)
		{
			printf("convert_image_with_letterbox fail! ret=%d\n", ret);
			return -1;
		}
		// Set Input Data
		inputs[0].index = 0;
		inputs[0].type = RKNN_TENSOR_UINT8;
		inputs[0].fmt = RKNN_TENSOR_NHWC;
		inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
		inputs[0].buf = dst_img.virt_addr;

		ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
		if (ret < 0)
		{
			printf("rknn_input_set fail! ret=%d\n", ret);
			return -1;
		}

		// Run
		ret = rknn_run(app_ctx->rknn_ctx, nullptr);
		if (ret < 0)
		{
			printf("rknn_run fail! ret=%d\n", ret);
			return -1;
		}

		// Get Output
		memset(outputs, 0, sizeof(outputs));
		for (int i = 0; i < app_ctx->io_num.n_output; i++)
		{
			outputs[i].index = i;
			outputs[i].want_float = (!app_ctx->is_quant);
		}
		ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
		if (ret < 0)
		{
			printf("rknn_outputs_get fail! ret=%d\n", ret);
		}

		// Post Process
		post_process(outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

		// Remeber to release rknn output
		rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

		// 应该检查输入图像是否为空
		if (dst_img.virt_addr != NULL)
		{
			free(dst_img.virt_addr);
		}

		return ret;
	}

	int post_process(void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, object_detect_result_list* od_results)
	{

		rknn_output* _outputs = (rknn_output*)outputs;
		std::vector<float> filterBoxes;
		std::vector<float> objProbs;
		std::vector<int> classId;
		int validCount = 0;
		int stride = 0;
		int grid_h = 0;
		int grid_w = 0;
		int model_in_w = app_ctx->model_width;
		int model_in_h = app_ctx->model_height;

		memset(od_results, 0, sizeof(object_detect_result_list));


		for (int i = 0; i < 3; i++)
		{
			grid_h = app_ctx->output_attrs[i].dims[2];
			grid_w = app_ctx->output_attrs[i].dims[3];
			stride = model_in_h / grid_h;
			if (app_ctx->is_quant)
			{
				validCount += process_i8((int8_t*)_outputs[i].buf, (int*)anchor[i], grid_h, grid_w, model_in_h, model_in_w, stride, filterBoxes, objProbs,
					classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale);
			}
			else
			{
				validCount += process_fp32((float*)_outputs[i].buf, (int*)anchor[i], grid_h, grid_w, model_in_h, model_in_w, stride, filterBoxes, objProbs,
					classId, conf_threshold);
			}
		}

		// no object detect
		if (validCount <= 0)
		{
			//printf("no object detect.\n");
			return 0;
		}

		std::vector<int> indexArray;
		for (int i = 0; i < validCount; ++i)
		{
			indexArray.push_back(i);
		}

		quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

		std::set<int> class_set(std::begin(classId), std::end(classId));

		for (auto c : class_set)
		{
			nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
		}

		int last_count = 0;
		od_results->count = 0;

		/* box valid detect target */
		for (int i = 0; i < validCount; ++i)
		{
			if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
			{
				continue;
			}
			int n = indexArray[i];

			float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
			float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
			float x2 = x1 + filterBoxes[n * 4 + 2];
			float y2 = y1 + filterBoxes[n * 4 + 3];
			int id = classId[n];
			float obj_conf = objProbs[i];
			od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
			od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
			od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
			od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
			od_results->results[last_count].prop = obj_conf;
			od_results->results[last_count].cls_id = id;
			last_count++;
		}
		od_results->count = last_count;
		return 0;
	}

};

#endif //_RKNN_DEMO_H_