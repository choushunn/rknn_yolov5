#pragma once
#include <iostream>

unsigned char* load_data(FILE* fp, size_t ofst, size_t sz);   // �������������ļ��м�������
int readLines(const char* fileName, char* lines[], int max_line);   // ������������ȡ�ļ��еĶ����ı�
unsigned char* load_model(const char* filename, int* model_size);   // ��������������ģ������
char* readLine(FILE* fp, char* buffer, int* len);   // �������������ļ��ж�ȡһ���ı�
int loadLabelName(const char* locationFilename, char* label[], int obj_class_num);
