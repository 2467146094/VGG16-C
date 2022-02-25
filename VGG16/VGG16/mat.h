#pragma once
#include"cnn.h"

//��ά������ת180��
float**** matRorate180(float**** mat, nSize matSize, int num);

//��ά������ת180��
float*** matRotate180(float*** mat, nSize matSize);

//�������������ֵ���±�
int argMax(float* mat, int length);

//����һ����ά����
float*** generateMatrix(nSize matSize);

//��ӡ��ά����
void printMatrix(float** mat, nSize matSize);

//��ӡ��ά����
void printMatrix(float*** mat, nSize matSize);

//�ͷ���ά����ռ�
void freeMatrix(float*** mat, nSize matSize);

//�ͷ���ά����ռ�
void freeMatrix(float**** mat, nSize matSize, int num);

//�ͷ���ά����λ�ÿռ�
void freeMatrix(valueLocation*** matLocation, nSize Size);

//��Ե������󣬸߶�����addh���������addw
float*** matEdgeExpand(float*** mat, nSize matSize, int addh, int addw);

//����ά����չƽ��һά
float* flatten(float*** mat, nSize matSize);

//����ת��
float** matrixTranspose(float** mat, nSize matSize);

//��ά�������
float** matrixMultiply(float** mat1, nSize matSize1, float** mat2, nSize matSize2);

//softmax����������
float* softmax(float* mat, int length);

//�������
float*** conv(float*** mat, float**** kernal, nSize matSize, nSize kernalSize, int kernalNum, int stride, int padding, float* bias);

//��ά����relu�����
float*** relu(float*** mat, nSize matSize);

//һά����relu�����
float* relu(float* mat, int length);

//���ػ�
float*** maxPooling(float*** mat, nSize matSize, nSize poolSize, int stride, int padding, valueLocation*** loc);

//ƽ���ػ�
float*** averagePooling(float*** mat, nSize matSize, nSize poolSize, int stride, int padding);

//ȫ���Ӽ���		
float* fc(float** weight, nSize weightSize, float* mat, int matLength, float* bias);

//����㣨��ӳػ���ľ���㣩�ֲ��ݶȣ���Ҫ����ĸò�����ľֲ��ݶ�gradient���ò������ߴ磬��һ��ػ���ľֲ��ݶȣ��ػ��������ߴ磬ȡ���������ֵ��λ�ã�
void convLocalGradientBeforePooling(float*** gradient, nSize outputMatSize, float*** poolGradient, nSize poolOutputMatSize, valueLocation*** loc);


/*
����㣨���治�ӳػ��㣩�ֲ��ݶ�(Ҳ�������ں���Ӿ����ĳػ���)
��Ҫ�ò�ľֲ��ݶ�gradient���ò������ߴ�outputMatSize���ò������Ȧ��padding����һ��ľ����ľֲ��ݶ�nextGradient,
��һ�㼤��ǰ�����nextV����һ�������ߴ�nextOutputMatSize����һ��ľ����ȨֵnextKernalWeight��
��һ��ľ���˳ߴ�nextKernalSize,��һ�����ͨ����nextChannels����һ��ľ������nextStride
*/
void convLocalGradient(float*** gradient, nSize outputMatSize, int padding, float*** nextGradient, float*** nextV, nSize nextOutputMatSize, float**** nextKernalWeight, nSize nextKernalSize, int nextChannels, int nextStride);


//�����Ȩֵ����(��Ҫ���ľ���ˣ�����˳ߴ磬���ͨ�������������ľֲ��ݶ�,�ֲ��ݶȳߴ磨����һ�������ߴ磩����һ������v(������relu����)����һ������ֵ�����Լ��ߴ�(����һ�������ߴ�)����һ�����������Ȧ�������������ѧϰ��)
void updateConvWeight(float**** kernalWeight, nSize kernalSize, int outChannels, float*** gradient, nSize outputMatSize, float*** v, float*** lastY, nSize inputMatSize, int padding, int stride, float learningRate);

//ȫ���Ӳ�Ȩֵ����(��Ҫ��һ���Ȩֵ����Ȩֵ����ߴ磬��һ��ľֲ��ݶ�gradient,��һ������v����һ������y��ѧϰ��)
void updateFcWeight(float** weight, nSize weightSize, float* gradient, float* v, float* lastY, float learningRate);

