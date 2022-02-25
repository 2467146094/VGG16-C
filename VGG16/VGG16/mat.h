#pragma once
#include"cnn.h"

//四维矩阵旋转180°
float**** matRorate180(float**** mat, nSize matSize, int num);

//三维矩阵旋转180°
float*** matRotate180(float*** mat, nSize matSize);

//返回数组中最大值的下标
int argMax(float* mat, int length);

//生成一个三维矩阵
float*** generateMatrix(nSize matSize);

//打印二维矩阵；
void printMatrix(float** mat, nSize matSize);

//打印三维矩阵；
void printMatrix(float*** mat, nSize matSize);

//释放三维矩阵空间
void freeMatrix(float*** mat, nSize matSize);

//释放四维矩阵空间
void freeMatrix(float**** mat, nSize matSize, int num);

//释放三维矩阵位置空间
void freeMatrix(valueLocation*** matLocation, nSize Size);

//边缘扩充矩阵，高度扩充addh，宽度扩充addw
float*** matEdgeExpand(float*** mat, nSize matSize, int addh, int addw);

//把三维矩阵展平成一维
float* flatten(float*** mat, nSize matSize);

//矩阵转置
float** matrixTranspose(float** mat, nSize matSize);

//二维矩阵相乘
float** matrixMultiply(float** mat1, nSize matSize1, float** mat2, nSize matSize2);

//softmax概率输出结果
float* softmax(float* mat, int length);

//卷积运算
float*** conv(float*** mat, float**** kernal, nSize matSize, nSize kernalSize, int kernalNum, int stride, int padding, float* bias);

//三维矩阵relu激活函数
float*** relu(float*** mat, nSize matSize);

//一维矩阵relu激活函数
float* relu(float* mat, int length);

//最大池化
float*** maxPooling(float*** mat, nSize matSize, nSize poolSize, int stride, int padding, valueLocation*** loc);

//平均池化
float*** averagePooling(float*** mat, nSize matSize, nSize poolSize, int stride, int padding);

//全连接计算		
float* fc(float** weight, nSize weightSize, float* mat, int matLength, float* bias);

//卷积层（后接池化层的卷积层）局部梯度（需要计算的该层输出的局部梯度gradient，该层的输出尺寸，后一层池化层的局部梯度，池化层的输出尺寸，取卷积层的最大值的位置）
void convLocalGradientBeforePooling(float*** gradient, nSize outputMatSize, float*** poolGradient, nSize poolOutputMatSize, valueLocation*** loc);


/*
卷积层（后面不接池化层）局部梯度(也可以用于后面接卷积层的池化层)
需要该层的局部梯度gradient，该层的输出尺寸outputMatSize，该层的扩充圈数padding，后一层的卷积层的局部梯度nextGradient,
后一层激活前的输出nextV，后一层的输出尺寸nextOutputMatSize，后一层的卷积核权值nextKernalWeight，
后一层的卷积核尺寸nextKernalSize,后一层输出通道数nextChannels，后一层的卷积步长nextStride
*/
void convLocalGradient(float*** gradient, nSize outputMatSize, int padding, float*** nextGradient, float*** nextV, nSize nextOutputMatSize, float**** nextKernalWeight, nSize nextKernalSize, int nextChannels, int nextStride);


//卷积层权值更新(需要这层的卷积核，卷积核尺寸，输出通道数，这层输出的局部梯度,局部梯度尺寸（即这一层的输出尺寸），这一层的输出v(用于求导relu激活)，上一层的输出值矩阵以及尺寸(即这一层的输入尺寸)和这一层卷积层的扩充圈数，卷积步长，学习率)
void updateConvWeight(float**** kernalWeight, nSize kernalSize, int outChannels, float*** gradient, nSize outputMatSize, float*** v, float*** lastY, nSize inputMatSize, int padding, int stride, float learningRate);

//全连接层权值更新(需要这一层的权值矩阵，权值矩阵尺寸，这一层的局部梯度gradient,这一层的输出v，上一层的输出y，学习率)
void updateFcWeight(float** weight, nSize weightSize, float* gradient, float* v, float* lastY, float learningRate);

