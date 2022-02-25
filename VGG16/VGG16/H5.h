#pragma once
#include"cnn.h"

//读取一维数据：偏置
void readWeight(const char* HDF5filename, const char* datasetName, float* bias);

//读取四维数据：卷积核权值
void readWeight(const char* HDF5filename, const char* datasetName, nSize kernalSize, int outChannels, float**** kernalWeight);

//读取二维数据：全连接层权值
void readWeight(const char* HDF5filename, const char* datasetName, nSize weightSize, float** weight);