#pragma once

#include<iostream>
using std::string;

//特征尺寸
typedef struct {
	int h;//高度
	int w;//宽度
	int c;//通道数channel
}nSize;

//最大池化层选取的值在原先的矩阵中的位置
typedef struct {
	int row;
	int column;
	int deep;
}valueLocation;

// 卷积层ConvLayer
typedef struct convolution_layer {
	nSize inputMatSize;  //输入图形的尺寸
	nSize outputMatSize; //图形在这层经过卷积后的输出尺寸

	int stride;//卷积核的步长
	int padding;//输入矩阵边界扩充的圈数

	nSize kernalSize; //卷积核尺寸 
	int outChannels;  //卷积核数量（组数），也即输出矩阵的通道数

	float**** kernalWeight;     //存放卷积核的权值，四维outChannels * kernalSize.c * kernalSize.h * kernalSize.w

	float* bias;   //偏置，偏置的大小，长度为outChannels,表示每一组卷积核有一个偏置

	float*** v; // 进入激活函数前的输入值,即卷积运算之后的值
	float*** y; // 经过激活函数后的输出值,用于下一层的输入

	float*** gradient; // 网络的局部梯度,δ值，形状为outputMatSize
}ConvLayer;


// 池化层 pooling
typedef struct pooling_layer {
	nSize inputMatSize;  //输入图形的尺寸
	nSize outputMatSize;   //池化后的图形尺寸

	int stride;//池化窗口的步长
	int padding;//输入矩阵边界扩充的圈数

	nSize poolSize;  //池化窗口大小

	valueLocation*** loc;//尺寸为outputMatSize

	int poolType;     //Pooling的方法,0表示最大池化，1表示平均池化

	float*** y; // 池化后矩阵的输出,用于下一层的输入
	float*** gradient; // 网络的局部梯度δ值，形状为outputMatSize
}PoolLayer;


// 全连接层
typedef struct fc_layer {
	int inputNum;   //输入数据的数目
	int outputNum;  //输出数据的数目
	float* bias;    //偏置

	float** weight; // 权重数据
	nSize weightSize;// 权重矩阵尺寸，大小为outputNum*inputNum

	float* v; // 进入激活函数前的输入值,即全连接运算后的值
	float* y; // 激活函数后神经元的输出,用于下一层的输入

	float* gradient; // 网络的局部梯度,δ值，长度为outputNum


}FCLayer;


//VGG16模型
typedef struct vgg16_network {
	//Block1
	ConvLayer* conv1;
	ConvLayer* conv2;
	PoolLayer* pool1;

	//Block2
	ConvLayer* conv3;
	ConvLayer* conv4;
	PoolLayer* pool2;

	//Block3
	ConvLayer* conv5;
	ConvLayer* conv6;
	ConvLayer* conv7;
	PoolLayer* pool3;

	//Block4
	ConvLayer* conv8;
	ConvLayer* conv9;
	ConvLayer* conv10;
	PoolLayer* pool4;

	//Block5
	ConvLayer* conv11;
	ConvLayer* conv12;
	ConvLayer* conv13;
	PoolLayer* pool5;

	//
	FCLayer* fc1;
	FCLayer* fc2;
	FCLayer* fc3;

}VGG16;


//初始化一个卷积层
ConvLayer* initConvLayer(nSize inputMatSize, nSize kernalSize, int outChannels, int stride, int padding, const char* HDF5filename, const char* weightDatasetName, const char* biasDatasetName);

//初始化一个池化层,poolType=0表示最大池化，1表示平均池化
PoolLayer* initPoolLayer(nSize inputMat, nSize poolSize, int poolType, int stride, int padding);

//初始化一个全连接层
FCLayer* initFCLayer(int inputNum, int outputNum, const char* HDF5filename, const char* weightDatasetName, const char* biasDatasetName);

//初始化一个VGG16模型
VGG16* initVGG16();

//调用VGG16模型
string inferenceVGG16(VGG16* vgg16, float*** mat, nSize matSize);

//训练VGG16模型
void trainingVGG16(VGG16* vgg16, float*** mat, nSize matSize, int label, float learningRate, string loss = "MeanSquaredError");