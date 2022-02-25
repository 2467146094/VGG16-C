#pragma once

#include<iostream>
using std::string;

//�����ߴ�
typedef struct {
	int h;//�߶�
	int w;//���
	int c;//ͨ����channel
}nSize;

//���ػ���ѡȡ��ֵ��ԭ�ȵľ����е�λ��
typedef struct {
	int row;
	int column;
	int deep;
}valueLocation;

// �����ConvLayer
typedef struct convolution_layer {
	nSize inputMatSize;  //����ͼ�εĳߴ�
	nSize outputMatSize; //ͼ������㾭������������ߴ�

	int stride;//����˵Ĳ���
	int padding;//�������߽������Ȧ��

	nSize kernalSize; //����˳ߴ� 
	int outChannels;  //�������������������Ҳ����������ͨ����

	float**** kernalWeight;     //��ž���˵�Ȩֵ����άoutChannels * kernalSize.c * kernalSize.h * kernalSize.w

	float* bias;   //ƫ�ã�ƫ�õĴ�С������ΪoutChannels,��ʾÿһ��������һ��ƫ��

	float*** v; // ���뼤���ǰ������ֵ,���������֮���ֵ
	float*** y; // ���������������ֵ,������һ�������

	float*** gradient; // ����ľֲ��ݶ�,��ֵ����״ΪoutputMatSize
}ConvLayer;


// �ػ��� pooling
typedef struct pooling_layer {
	nSize inputMatSize;  //����ͼ�εĳߴ�
	nSize outputMatSize;   //�ػ����ͼ�γߴ�

	int stride;//�ػ����ڵĲ���
	int padding;//�������߽������Ȧ��

	nSize poolSize;  //�ػ����ڴ�С

	valueLocation*** loc;//�ߴ�ΪoutputMatSize

	int poolType;     //Pooling�ķ���,0��ʾ���ػ���1��ʾƽ���ػ�

	float*** y; // �ػ����������,������һ�������
	float*** gradient; // ����ľֲ��ݶȦ�ֵ����״ΪoutputMatSize
}PoolLayer;


// ȫ���Ӳ�
typedef struct fc_layer {
	int inputNum;   //�������ݵ���Ŀ
	int outputNum;  //������ݵ���Ŀ
	float* bias;    //ƫ��

	float** weight; // Ȩ������
	nSize weightSize;// Ȩ�ؾ���ߴ磬��СΪoutputNum*inputNum

	float* v; // ���뼤���ǰ������ֵ,��ȫ����������ֵ
	float* y; // ���������Ԫ�����,������һ�������

	float* gradient; // ����ľֲ��ݶ�,��ֵ������ΪoutputNum


}FCLayer;


//VGG16ģ��
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


//��ʼ��һ�������
ConvLayer* initConvLayer(nSize inputMatSize, nSize kernalSize, int outChannels, int stride, int padding, const char* HDF5filename, const char* weightDatasetName, const char* biasDatasetName);

//��ʼ��һ���ػ���,poolType=0��ʾ���ػ���1��ʾƽ���ػ�
PoolLayer* initPoolLayer(nSize inputMat, nSize poolSize, int poolType, int stride, int padding);

//��ʼ��һ��ȫ���Ӳ�
FCLayer* initFCLayer(int inputNum, int outputNum, const char* HDF5filename, const char* weightDatasetName, const char* biasDatasetName);

//��ʼ��һ��VGG16ģ��
VGG16* initVGG16();

//����VGG16ģ��
string inferenceVGG16(VGG16* vgg16, float*** mat, nSize matSize);

//ѵ��VGG16ģ��
void trainingVGG16(VGG16* vgg16, float*** mat, nSize matSize, int label, float learningRate, string loss = "MeanSquaredError");