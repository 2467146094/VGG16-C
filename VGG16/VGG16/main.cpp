#include<opencv2/opencv.hpp>
#include"hdf5.h"
#include<iostream>
#include"mat.h"
#include"cnn.h"
#include"image.h"
#include"H5.h"

using namespace cv;
using namespace std;

int main(void) {

	//��ȡͼƬ����ͼƬ����ת��������
	//const char* imageAddr = "testImage/car1_751.png";
	//const char* imageAddr = "testImage/car2_817.png";
	//const char* imageAddr = "testImage/car3_817.png";
	//const char* imageAddr = "testImage/cat1_281.png";
	//const char* imageAddr = "testImage/cat2_876.png";
	//const char* imageAddr = "testImage/cat3_283.png";
	//const char* imageAddr = "testImage/dog1_207.png";
	//const char* imageAddr = "testImage/dog2_258.png";
	//const char* imageAddr = "testImage/dog3_207.png";
	//const char* imageAddr = "testImage/flower.png";
	const char* imageAddr = "testImage/bird.png";
	
	

	nSize matSize = { 224,224,3 };
	float*** mat = imgToMatrix(imageAddr, matSize);

	//��ʼ��VGG16ģ��
	VGG16* vgg16 = initVGG16();



	//����ģ��
	string label = inferenceVGG16(vgg16, mat, matSize);
	
	//��ʾͼƬ
	Mat img = imread(imageAddr);
	showImg("image", img, label);
	waitKey();


	////ѵ��ģ��
	//trainingVGG16(vgg16, mat, matSize, 100, 0.2, "MeanSquaredError");

	return 0;
}