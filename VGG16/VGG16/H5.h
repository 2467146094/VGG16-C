#pragma once
#include"cnn.h"

//��ȡһά���ݣ�ƫ��
void readWeight(const char* HDF5filename, const char* datasetName, float* bias);

//��ȡ��ά���ݣ������Ȩֵ
void readWeight(const char* HDF5filename, const char* datasetName, nSize kernalSize, int outChannels, float**** kernalWeight);

//��ȡ��ά���ݣ�ȫ���Ӳ�Ȩֵ
void readWeight(const char* HDF5filename, const char* datasetName, nSize weightSize, float** weight);