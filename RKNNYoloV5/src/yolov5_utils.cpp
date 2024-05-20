#include "yolov5_utils.h"
/**
 * @brief ���ļ��м�������
 *
 * @param fp �ļ�ָ��
 * @param ofst �������ļ��е�ƫ����
 * @param sz Ҫ���ص����ݴ�С
 * @return unsigned char* ���ؼ��ص�����ָ�룬����ʧ�ܷ��� NULL
 */
unsigned char* load_data(FILE* fp, size_t ofst, size_t sz) {
	unsigned char* data;    // ����ָ��
	int ret;                // ����ֵ

	data = NULL;            // ��ʼ������ָ��Ϊ NULL

	// ����ļ�ָ���Ƿ�Ϊ NULL
	if (NULL == fp) {
		return NULL;
	}

	// ���ļ�ָ���ƶ���ָ����ƫ����λ��
	ret = fseek(fp, ofst, SEEK_SET);
	if (ret != 0) {
		printf("blob seek failure.\n");
		return NULL;
	}

	// �����ڴ����ڴ洢����
	data = (unsigned char*)malloc(sz);
	if (data == NULL) {
		printf("buffer malloc failure.\n");
		return NULL;
	}

	// ���ļ��ж�ȡ���ݵ��ڴ���
	ret = fread(data, 1, sz, fp);

	// ���ؼ��ص�����ָ��
	return data;
}
/**
 * @brief ���ļ��ж�ȡһ������
 *
 * @param fp �ļ�ָ��
 * @param buffer �洢��ȡ���ݵĻ�����
 * @param len ��ȡ���ݵĳ���
 * @return char* ��ȡ�����ݣ�����������󷵻� NULL
 */
char* readLine(FILE* fp, char* buffer, int* len) {
	int ch;
	int i = 0;
	size_t buff_len = 0;

	// ��ʼ��������
	buffer = (char*)malloc(buff_len + 1);
	if (!buffer)
		return NULL; // �ڴ治��

	// ���ַ���ȡֱ�����з����ļ�������
	while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
		buff_len++;

		// ��̬��չ������
		void* tmp = realloc(buffer, buff_len + 1);
		if (tmp == NULL) {
			free(buffer);
			return NULL; // �ڴ治��
		}
		buffer = (char*)tmp;

		// ���ַ���ӵ�������
		buffer[i] = (char)ch;
		i++;
	}
	buffer[i] = '\0'; // ����ַ���������

	*len = buff_len; // �洢��ȡ���ݵĳ���

	// ����Ƿ񵽴��ļ���β
	if (ch == EOF && (i == 0 || ferror(fp))) {
		free(buffer);
		return NULL; // �ļ���ȡ����
	}

	return buffer;
}

/**
 * @brief ���ļ��ж�ȡ��������
 *
 * @param fileName �ļ���
 * @param lines �洢��ȡ���ݵ�����
 * @param max_line �������
 * @return int ��ȡ������������������󷵻� -1
 */
int readLines(const char* fileName, char* lines[], int max_line) {
	FILE* file = fopen(fileName, "r"); // ���ļ�
	char* s;
	int   i = 0;
	int   n = 0;

	if (file == NULL) {
		printf("Open %s fail!\n", fileName);
		return -1; // ���ļ�ʧ��
	}

	// ���ж�ȡ�ļ�����
	while ((s = readLine(file, s, &n)) != NULL) {
		lines[i++] = s; // ����ȡ���д洢��������
		if (i >= max_line)
			break; // �ﵽ�������ʱ�˳�ѭ��
	}
	fclose(file); // �ر��ļ�
	return i; // ���ض�ȡ������
}


/**
 * @brief ���ļ��м���ģ������
 *
 * @param filename ģ���ļ���
 * @param model_size ���ڴ洢ģ�ʹ�С��ָ��
 * @return unsigned char* ����ģ�����ݵ�ָ�룬����ʧ�ܷ��� NULL
 */
unsigned char* load_model(const char* filename, int* model_size) {
	FILE* fp;               // �ļ�ָ��
	unsigned char* data;   // ģ������ָ��

	// ��ģ���ļ�
	fp = fopen(filename, "rb");
	if (NULL == fp) {
		printf("Open file %s failed.\n", filename);
		return NULL;
	}

	// ��ȡģ���ļ���С
	fseek(fp, 0, SEEK_END);
	int size = ftell(fp);

	// ����ģ������
	data = load_data(fp, 0, size);

	// �ر��ļ�
	fclose(fp);

	// ��ģ�ʹ�С�洢�� model_size ָ����
	*model_size = size;

	// ����ģ������ָ��
	return data;
}


/**
 * @brief ���ر�ǩ����
 *
 * @param locationFilename ��ǩ�ļ���
 * @param label �洢��ǩ���Ƶ�����
 * @return int ����ֵΪ 0 ��ʾ�ɹ�
 */
int loadLabelName(const char* locationFilename, char* label[], int obj_class_num) {
	printf("loadLabelName %s\n", locationFilename); // ��ӡ���ر�ǩ���Ƶ���Ϣ
	readLines(locationFilename, label, obj_class_num); // ���� readLines ������ȡ��ǩ����
	return 0; // ���ؼ��سɹ�
}