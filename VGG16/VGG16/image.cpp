#include"image.h"

using namespace cv;
using namespace std;

//把图片转换成数组
float*** imgToMatrix(const char* filename, nSize imgSize) {
	Mat img = imread(filename, IMREAD_COLOR | IMREAD_ANYDEPTH);
	//把图片转换成224*224形式
	if (img.rows != 224 || img.cols != 224) {
		resize(img, img, Size(224, 224));
	}

	//申请数组空间
	float*** mat = (float***)malloc(imgSize.c * sizeof(float**));
	for (int c = 0; c < imgSize.c; c++) {
		mat[c] = (float**)malloc(imgSize.h * sizeof(float*));
		for (int h = 0; h < imgSize.h; h++) {
			mat[c][h] = (float*)malloc(imgSize.w * sizeof(float));
		}
	}

	//把图片的像素写入数组
	for (int h = 0; h < imgSize.h; h++) {
		for (int w = 0; w < imgSize.w; w++) {
			Vec3b vc3 = img.at<Vec3b>(h, w);
			for (int c = 0; c < imgSize.c; c++) {
				mat[c][h][w] = (float)vc3.val[c];//像素值（0，255）
			}
		}
	}

	return mat;
}

//显示图片，添加上文字
void showImg(string windowName, Mat mat, string text) {
	putText(mat, text, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 225));
	imshow(windowName, mat);
}