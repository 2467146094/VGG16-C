#pragma once

#include<opencv2/opencv.hpp>
#include"cnn.h"

using std::string;
using cv::Mat;

//把图片转换成数组
float*** imgToMatrix(const char* filename, nSize imgSize);

//显示图片，添加上文字
void showImg(string windowName, Mat mat, string text);