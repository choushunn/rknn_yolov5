#pragma once
#include <iostream>

unsigned char* load_data(FILE* fp, size_t ofst, size_t sz);   // 函数声明：从文件中加载数据
int readLines(const char* fileName, char* lines[], int max_line);   // 函数声明：读取文件中的多行文本
unsigned char* load_model(const char* filename, int* model_size);   // 函数声明：加载模型数据
char* readLine(FILE* fp, char* buffer, int* len);   // 函数声明：从文件中读取一行文本
int loadLabelName(const char* locationFilename, char* label[], int obj_class_num);
