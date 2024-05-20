#include "yolov5_utils.h"
/**
 * @brief 从文件中加载数据
 *
 * @param fp 文件指针
 * @param ofst 数据在文件中的偏移量
 * @param sz 要加载的数据大小
 * @return unsigned char* 返回加载的数据指针，加载失败返回 NULL
 */
unsigned char* load_data(FILE* fp, size_t ofst, size_t sz) {
	unsigned char* data;    // 数据指针
	int ret;                // 返回值

	data = NULL;            // 初始化数据指针为 NULL

	// 检查文件指针是否为 NULL
	if (NULL == fp) {
		return NULL;
	}

	// 将文件指针移动到指定的偏移量位置
	ret = fseek(fp, ofst, SEEK_SET);
	if (ret != 0) {
		printf("blob seek failure.\n");
		return NULL;
	}

	// 分配内存用于存储数据
	data = (unsigned char*)malloc(sz);
	if (data == NULL) {
		printf("buffer malloc failure.\n");
		return NULL;
	}

	// 从文件中读取数据到内存中
	ret = fread(data, 1, sz, fp);

	// 返回加载的数据指针
	return data;
}
/**
 * @brief 从文件中读取一行内容
 *
 * @param fp 文件指针
 * @param buffer 存储读取内容的缓冲区
 * @param len 读取内容的长度
 * @return char* 读取的内容，如果发生错误返回 NULL
 */
char* readLine(FILE* fp, char* buffer, int* len) {
	int ch;
	int i = 0;
	size_t buff_len = 0;

	// 初始化缓冲区
	buffer = (char*)malloc(buff_len + 1);
	if (!buffer)
		return NULL; // 内存不足

	// 逐字符读取直到换行符或文件结束符
	while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
		buff_len++;

		// 动态扩展缓冲区
		void* tmp = realloc(buffer, buff_len + 1);
		if (tmp == NULL) {
			free(buffer);
			return NULL; // 内存不足
		}
		buffer = (char*)tmp;

		// 将字符添加到缓冲区
		buffer[i] = (char)ch;
		i++;
	}
	buffer[i] = '\0'; // 添加字符串结束符

	*len = buff_len; // 存储读取内容的长度

	// 检测是否到达文件结尾
	if (ch == EOF && (i == 0 || ferror(fp))) {
		free(buffer);
		return NULL; // 文件读取错误
	}

	return buffer;
}

/**
 * @brief 从文件中读取多行内容
 *
 * @param fileName 文件名
 * @param lines 存储读取内容的数组
 * @param max_line 最大行数
 * @return int 读取的行数，如果发生错误返回 -1
 */
int readLines(const char* fileName, char* lines[], int max_line) {
	FILE* file = fopen(fileName, "r"); // 打开文件
	char* s;
	int   i = 0;
	int   n = 0;

	if (file == NULL) {
		printf("Open %s fail!\n", fileName);
		return -1; // 打开文件失败
	}

	// 逐行读取文件内容
	while ((s = readLine(file, s, &n)) != NULL) {
		lines[i++] = s; // 将读取的行存储到数组中
		if (i >= max_line)
			break; // 达到最大行数时退出循环
	}
	fclose(file); // 关闭文件
	return i; // 返回读取的行数
}


/**
 * @brief 从文件中加载模型数据
 *
 * @param filename 模型文件名
 * @param model_size 用于存储模型大小的指针
 * @return unsigned char* 返回模型数据的指针，加载失败返回 NULL
 */
unsigned char* load_model(const char* filename, int* model_size) {
	FILE* fp;               // 文件指针
	unsigned char* data;   // 模型数据指针

	// 打开模型文件
	fp = fopen(filename, "rb");
	if (NULL == fp) {
		printf("Open file %s failed.\n", filename);
		return NULL;
	}

	// 获取模型文件大小
	fseek(fp, 0, SEEK_END);
	int size = ftell(fp);

	// 加载模型数据
	data = load_data(fp, 0, size);

	// 关闭文件
	fclose(fp);

	// 将模型大小存储到 model_size 指针中
	*model_size = size;

	// 返回模型数据指针
	return data;
}


/**
 * @brief 加载标签名称
 *
 * @param locationFilename 标签文件名
 * @param label 存储标签名称的数组
 * @return int 返回值为 0 表示成功
 */
int loadLabelName(const char* locationFilename, char* label[], int obj_class_num) {
	printf("loadLabelName %s\n", locationFilename); // 打印加载标签名称的信息
	readLines(locationFilename, label, obj_class_num); // 调用 readLines 函数读取标签名称
	return 0; // 返回加载成功
}