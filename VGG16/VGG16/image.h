#pragma once

#include<opencv2/opencv.hpp>
#include"cnn.h"

using std::string;
using cv::Mat;

//��ͼƬת��������
float*** imgToMatrix(const char* filename, nSize imgSize);

//��ʾͼƬ�����������
void showImg(string windowName, Mat mat, string text);