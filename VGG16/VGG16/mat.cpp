#include<iostream>
#include<string>
#include"mat.h"

using namespace std;


//四维矩阵旋转180°
float**** matRorate180(float**** mat, nSize matSize, int num) {
	float**** rotateMat = (float****)malloc(num * sizeof(float***));
	for (int n = 0; n < num; n++) {
		rotateMat[n] = (float***)malloc(matSize.c * sizeof(float**));
		for (int c = 0; c < matSize.c; c++) {
			rotateMat[n][c] = (float**)malloc(matSize.h * sizeof(float*));
			for (int h = 0; h < matSize.h; h++) {
				rotateMat[n][c][h] = (float*)malloc(matSize.w * sizeof(float));
			}
		}
	}

	for (int n = 0; n < num; n++) {
		for (int c = 0; c < matSize.c; c++) {
			for (int h = 0; h < matSize.h; h++) {
				for (int w = 0; w < matSize.w; w++) {
					rotateMat[n][c][h][w] = mat[n][c][matSize.h - h - 1][matSize.w - w - 1];
				}
			}
		}
	}

	return rotateMat;
}

//三维矩阵旋转180°
float*** matRotate180(float*** mat, nSize matSize) {
	float*** rotateMat = (float***)malloc(matSize.c * sizeof(float**));
	for (int c = 0; c < matSize.c; c++) {
		rotateMat[c] = (float**)malloc(matSize.h * sizeof(float*));
		for (int h = 0; h < matSize.h; h++) {
			rotateMat[c][h] = (float*)malloc(matSize.w * sizeof(float));
		}
	}


	for (int c = 0; c < matSize.c; c++) {
		for (int h = 0; h < matSize.h; h++) {
			for (int w = 0; w < matSize.w; w++) {
				rotateMat[c][h][w] = mat[c][matSize.h - h - 1][matSize.w - w - 1];
			}
		}
	}

	return rotateMat;
}

//返回数组中最大值的下标
int argMax(float* mat, int length) {
	int loc = 0;
	float max = FLT_MIN;
	for (int i = 0; i < length; i++) {
		if (mat[i] > max) {
			max = mat[i];
			loc = i;
		}
	}

	return loc;
}


//生成一个三维矩阵
float*** generateMatrix(nSize matSize) {
	srand((unsigned)time(NULL));
	float*** mat = (float***)malloc(matSize.c * sizeof(float**));
	for (int c = 0; c < matSize.c; c++) {
		mat[c] = (float**)malloc(matSize.h * sizeof(float*));
		for (int h = 0; h < matSize.h; h++) {
			mat[c][h] = (float*)malloc(matSize.w * sizeof(float));
			for (int w = 0; w < matSize.w; w++) {
				mat[c][h][w] = rand() % 3 - 1;
			}
		}
	}

	return mat;
}

//打印二维矩阵；
void printMatrix(float** mat, nSize matSize) {
	for (int h = 0; h < matSize.h; h++) {
		for (int w = 0; w < matSize.w; w++) {
			cout << mat[h][w] << "  ";
		}
		cout << endl;
	}
}


//打印三维矩阵；
void printMatrix(float*** mat, nSize matSize) {
	for (int c = 0; c < matSize.c; c++) {
		printf("channel(%d):\n", c);
		for (int h = 0; h < matSize.h; h++) {
			for (int w = 0; w < matSize.w; w++) {
				printf("%.2f  ", mat[c][h][w]);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n\n");
}


//释放三维矩阵空间
void freeMatrix(float*** mat, nSize matSize) {
	for (int c = 0; c < matSize.c; c++) {
		for (int h = 0; h < matSize.h; h++) {
			free(mat[c][h]);
		}
	}

	for (int c = 0; c < matSize.c; c++) {
		free(mat[c]);
	}

	free(mat);
}


//释放四维矩阵空间
void freeMatrix(float**** mat, nSize matSize, int num) {
	for (int n = 0; n < num; n++) {
		for (int c = 0; c < matSize.c; c++) {
			for (int h = 0; h < matSize.h; h++) {
				free(mat[n][c][h]);
			}
		}
	}

	for (int n = 0; n < num; n++) {
		for (int c = 0; c < matSize.c; c++) {
			free(mat[n][c]);
		}
	}

	for (int n = 0; n < num; n++) {
		free(mat[n]);
	}

	free(mat);
}


//释放三维矩阵位置空间
void freeMatrix(valueLocation*** matLocation, nSize Size) {
	for (int c = 0; c < Size.c; c++) {
		for (int h = 0; h < Size.h; h++) {
			free(matLocation[c][h]);
		}
	}

	for (int c = 0; c < Size.c; c++) {
		free(matLocation[c]);
	}

	free(matLocation);
}



//边缘扩充矩阵，高度扩充addh行，宽度扩充addw列
float*** matEdgeExpand(float*** mat, nSize matSize, int addh, int addw) {
	int c, h, w;
	int outSizeH = matSize.h + addh;
	int outSizeW = matSize.w + addw;
	int outSizeC = matSize.c;

	//矩阵上下左右分别要加的行数或列数，对半分，分奇偶情况
	int upperLine = (addh + (addh % 2)) / 2;
	int lowLine = addh - upperLine;
	int leftColumn = (addw + (addw % 2)) / 2;
	int rightColumn = addw - leftColumn;

	//给矩阵分配内存
	float*** outMat = (float***)malloc(outSizeC * sizeof(float**));
	for (c = 0; c < matSize.c; c++) {
		outMat[c] = (float**)malloc(outSizeH * sizeof(float*));
		for (h = 0; h < outSizeH; h++) {
			outMat[c][h] = (float*)malloc(outSizeW * sizeof(float));
		}
	}

	//给矩阵赋值
	for (c = 0; c < outSizeC; c++) {
		for (h = 0; h < outSizeH; h++) {
			for (w = 0; w < outSizeW; w++) {
				if (h < upperLine || h>(outSizeH - lowLine - 1) || w<leftColumn || w>(outSizeW - rightColumn - 1)) {
					outMat[c][h][w] = float(0);
				}
				else {
					outMat[c][h][w] = mat[c][h - upperLine][w - leftColumn];
				}
			}
		}
	}

	return outMat;
}


//卷积运算,padding表示填充几圈
float*** conv(float*** mat, float**** kernal, nSize matSize, nSize kernalSize, int kernalNum, int stride, int padding, float* bias) {
	int c, h, w, i, j, k;
	int addh = 2 * padding, addw = 2 * padding;//增加的行数和列数

	//计算输出矩阵的尺寸
	int outSizeC = kernalNum;
	int outSizeH = (matSize.h + addh - kernalSize.h) / stride + 1;
	int outSizeW = (matSize.w + addw - kernalSize.w) / stride + 1;

	//定义输出矩阵
	float*** outMat = (float***)malloc(outSizeC * sizeof(float**));
	for (c = 0; c < outSizeC; c++) {
		outMat[c] = (float**)malloc(outSizeH * sizeof(float*));
		for (h = 0; h < outSizeH; h++) {
			outMat[c][h] = (float*)malloc(outSizeW * sizeof(float));
		}
	}

	//如果需要扩充矩阵
	if (padding > 0) {
		float*** expandMat = matEdgeExpand(mat, matSize, addh, addw);

		//卷积计算
		for (c = 0; c < outSizeC; c++) {
			for (h = 0; h < outSizeH; h++) {
				for (w = 0; w < outSizeW; w++) {
					float sum = bias[c];
					for (i = 0; i < kernalSize.c; i++) {
						for (j = 0; j < kernalSize.h; j++) {
							for (k = 0; k < kernalSize.w; k++) {
								sum += kernal[c][i][j][k] * expandMat[i][h * stride + j][w * stride + k];
							}
						}
					}
					outMat[c][h][w] = sum;
				}
			}
		}

		//释放expandMat
		for (c = 0; c < matSize.c; c++) {
			for (h = 0; h < matSize.h + addh; h++) {
				free(expandMat[c][h]);
			}
		}
		for (c = 0; c < matSize.c; c++) {
			free(expandMat[c]);
		}
		free(expandMat);
	}

	//不需要扩充矩阵
	else {
		//卷积计算
		for (c = 0; c < outSizeC; c++) {
			for (h = 0; h < outSizeH; h++) {
				for (w = 0; w < outSizeW; w++) {
					float sum = bias[c];
					for (i = 0; i < kernalSize.c; i++) {
						for (j = 0; j < kernalSize.h; j++) {
							for (k = 0; k < kernalSize.w; k++) {
								sum += kernal[c][i][j][k] * mat[i][h * stride + j][w * stride + k];
							}
						}
					}
					outMat[c][h][w] = sum;
				}
			}
		}
	}

	return outMat;
}


//relu激活函数
float*** relu(float*** mat, nSize matSize) {
	int outSizeC = matSize.c;
	int outSizeH = matSize.h;
	int outSizeW = matSize.w;
	int c, h, w;

	float*** outMat = (float***)malloc(outSizeC * sizeof(float**));
	for (c = 0; c < outSizeC; c++) {
		outMat[c] = (float**)malloc(outSizeH * sizeof(float*));
		for (h = 0; h < outSizeH; h++) {
			outMat[c][h] = (float*)malloc(outSizeW * sizeof(float));
		}
	}

	for (c = 0; c < outSizeC; c++) {
		for (h = 0; h < outSizeH; h++) {
			for (w = 0; w < outSizeW; w++) {
				outMat[c][h][w] = mat[c][h][w] > 0 ? mat[c][h][w] : 0;
			}
		}
	}

	return outMat;
}


//一维矩阵relu激活函数
float* relu(float* mat, int length) {
	float* outMat = (float*)malloc(length * sizeof(float));
	for (int i = 0; i < length; i++) {
		outMat[i] = mat[i] > 0 ? mat[i] : 0;
	}

	return outMat;
}


//最大池化
float*** maxPooling(float*** mat, nSize matSize, nSize poolSize, int stride, int padding, valueLocation*** loc) {
	int c, h, w, i, j;
	float max;
	int addh = 2 * padding, addw = 2 * padding;//增加的行数和列数

	//计算输出矩阵的尺寸
	int outSizeC = matSize.c;
	int outSizeH = (matSize.h + addh - poolSize.h) / stride + 1;
	int outSizeW = (matSize.w + addw - poolSize.w) / stride + 1;

	//定义输出矩阵
	float*** outMat = (float***)malloc(outSizeC * sizeof(float**));
	for (c = 0; c < outSizeC; c++) {
		outMat[c] = (float**)malloc(outSizeH * sizeof(float*));
		for (h = 0; h < outSizeH; h++) {
			outMat[c][h] = (float*)malloc(outSizeW * sizeof(float));
		}
	}

	//如果需要扩充矩阵
	if (padding > 0) {
		float*** expandMat = matEdgeExpand(mat, matSize, addh, addw);

		//最大池化计算
		for (c = 0; c < outSizeC; c++) {
			for (h = 0; h < outSizeH; h++) {
				for (w = 0; w < outSizeW; w++) {
					max = expandMat[c][h * stride][w * stride];
					loc[c][h][w].deep = c;
					loc[c][h][w].row = h * stride;
					loc[c][h][w].column = w * stride;
					for (i = 0; i < poolSize.h; i++) {
						for (j = 0; j < poolSize.w; j++) {
							if (expandMat[c][h * stride + i][w * stride + j] > max) {
								//记录最大值
								max = expandMat[c][h * stride + i][w * stride + j];
								//记录下最大值的位置
								loc[c][h][w].deep = c;
								loc[c][h][w].row = h * stride + i;
								loc[c][h][w].column = w * stride + j;
							}
						}
					}
					outMat[c][h][w] = max;
				}
			}
		}

		//释放expandMat
		for (c = 0; c < matSize.c; c++) {
			for (h = 0; h < matSize.h + addh; h++) {
				free(expandMat[c][h]);
			}
		}
		for (c = 0; c < matSize.c; c++) {
			free(expandMat[c]);
		}
		free(expandMat);

	}
	//不需要扩充矩阵
	else {
		//最大池化计算
		for (c = 0; c < outSizeC; c++) {
			for (h = 0; h < outSizeH; h++) {
				for (w = 0; w < outSizeW; w++) {
					max = mat[c][h * stride][w * stride];
					loc[c][h][w].deep = c;
					loc[c][h][w].row = h * stride;
					loc[c][h][w].column = w * stride;
					for (i = 0; i < poolSize.h; i++) {
						for (j = 0; j < poolSize.w; j++) {
							if (mat[c][h * stride + i][w * stride + j] > max) {
								//记录最大值
								max = mat[c][h * stride + i][w * stride + j];
								//记录下最大值的位置
								loc[c][h][w].deep = c;
								loc[c][h][w].row = h * stride + i;
								loc[c][h][w].column = w * stride + j;
							}
						}
					}
					outMat[c][h][w] = max;
				}
			}
		}
	}

	return outMat;
}


//平均池化
float*** averagePooling(float*** mat, nSize matSize, nSize poolSize, int stride, int padding) {
	int c, h, w, i, j;
	int addh = 2 * padding, addw = 2 * padding;//增加的行数和列数	

	//计算输出矩阵的尺寸
	int outSizeC = matSize.c;
	int outSizeH = (matSize.h + addh - poolSize.h) / stride + 1;
	int outSizeW = (matSize.w + addw - poolSize.w) / stride + 1;

	//定义输出矩阵
	float*** outMat = (float***)malloc(outSizeC * sizeof(float**));
	for (c = 0; c < outSizeC; c++) {
		outMat[c] = (float**)malloc(outSizeH * sizeof(float*));
		for (h = 0; h < outSizeH; h++) {
			outMat[c][h] = (float*)malloc(outSizeW * sizeof(float));
		}
	}

	//如果需要扩充矩阵
	if (padding > 0) {
		float*** expandMat = matEdgeExpand(mat, matSize, addh, addw);
		//平均池化计算
		for (c = 0; c < outSizeC; c++) {
			for (h = 0; h < outSizeH; h++) {
				for (w = 0; w < outSizeW; w++) {
					float sum = 0;
					for (i = 0; i < poolSize.h; i++) {
						for (j = 0; j < poolSize.w; j++) {
							sum += expandMat[c][h * stride + i][w * stride + j];
						}
					}
					outMat[c][h][w] = sum / (poolSize.h * poolSize.w);
				}
			}
		}

		//释放expandMat
		for (c = 0; c < matSize.c; c++) {
			for (h = 0; h < matSize.h + addh; h++) {
				free(expandMat[c][h]);
			}
		}
		for (c = 0; c < matSize.c; c++) {
			free(expandMat[c]);
		}
		free(expandMat);

	}
	//不需要扩充矩阵
	else {
		//平均池化计算
		for (c = 0; c < outSizeC; c++) {
			for (h = 0; h < outSizeH; h++) {
				for (w = 0; w < outSizeW; w++) {
					float sum = 0;
					for (i = 0; i < poolSize.h; i++) {
						for (j = 0; j < poolSize.w; j++) {
							sum += mat[c][h * stride + i][w * stride + j];
						}
					}
					outMat[c][h][w] = sum / (poolSize.h * poolSize.w);
				}
			}
		}
	}

	return outMat;
}


////把三维矩阵展平成一维(从每一行每一行展开，c*h*w)
//float* flatten(float*** mat, nSize matSize) {
//	int len = matSize.c * matSize.h * matSize.w;
//
//	float* outMat = (float*)malloc(len * sizeof(float));
//
//	for (int c = 0; c < matSize.c; c++)
//	{
//		for (int h = 0; h < matSize.h; h++)
//		{
//			for (int w = 0; w < matSize.w; w++)
//			{
//				outMat[c * matSize.h * matSize.w + h * matSize.w + w] = mat[c][h][w];
//			}
//		}
//	}
//
//	return outMat;
//}


//把三维矩阵展平成一维（从深度开始展开，h*w*c）
float* flatten(float*** mat, nSize matSize) {
	int len = matSize.c * matSize.h * matSize.w;

	float* outMat = (float*)malloc(len * sizeof(float));

	for (int c = 0; c < matSize.c; c++)
	{
		for (int h = 0; h < matSize.h; h++)
		{
			for (int w = 0; w < matSize.w; w++)
			{
				outMat[h * matSize.w * matSize.c + w * matSize.c + c] = mat[c][h][w];
			}
		}
	}

	return outMat;
}


//全连接计算(weightSize.w=matLength)
float* fc(float** weight, nSize weightSize, float* mat, int matLength, float* bias) {
	int outLength = weightSize.h;

	float* result = (float*)malloc(outLength * sizeof(float));

	for (int i = 0; i < outLength; i++) {
		float sum = bias[i];
		for (int w = 0; w < weightSize.w; w++) {
			sum += weight[i][w] * mat[w];
		}
		result[i] = sum;
	}

	return result;
}


//矩阵转置
float** matrixTranspose(float** mat, nSize matSize) {
	int h, w;
	int outSizeH = matSize.w;
	int outSizeW = matSize.h;


	float** out = (float**)malloc(outSizeH * sizeof(float*));
	for (h = 0; h < outSizeH; h++) {
		out[h] = (float*)malloc(outSizeW * sizeof(float));
	}

	for (h = 0; h < outSizeH; h++) {
		for (w = 0; w < outSizeW; w++) {
			out[h][w] = mat[w][h];
		}
	}
	return out;
}


//二维矩阵相乘
float** matrixMultiply(float** mat1, nSize matSize1, float** mat2, nSize matSize2) {
	int outSizeH = matSize1.h;
	int outSizeW = matSize2.w;
	int h, w, i;

	float** out = (float**)malloc(outSizeH * sizeof(float*));
	for (h = 0; h < outSizeH; h++) {
		out[h] = (float*)malloc(outSizeW * sizeof(float));
		for (w = 0; w < outSizeW; w++) {
			out[h][w] = 0;
		}
	}

	for (h = 0; h < outSizeH; h++) {
		for (w = 0; w < outSizeW; w++) {
			float sum = 0;
			for (i = 0; i < matSize1.w; i++)
			{
				sum += mat1[h][i] * mat2[i][w];
			}
			out[h][w] = sum;
		}
	}
	return out;
}


//softmax概率输出结果
float* softmax(float* mat, int length) {
	float* out = (float*)malloc(length * sizeof(float));
	int i;
	float sum = 0;
	for (i = 0; i < length; i++) {
		sum += exp(mat[i]);
	}
	for (i = 0; i < length; i++) {
		out[i] = exp(mat[i]) / sum;
	}
	return out;
}


//卷积层（后接池化层的卷积层）局部梯度（需要计算的该层输出的局部梯度gradient，该层的输出尺寸，后一层池化层的局部梯度，池化层的输出尺寸，取卷积层的最大值的位置）
void convLocalGradientBeforePooling(float*** gradient, nSize outputMatSize, float*** poolGradient, nSize poolOutputMatSize, valueLocation*** loc) {
	for (int c = 0; c < outputMatSize.c; c++) {//初始化为0
		for (int h = 0; h < outputMatSize.h; h++) {
			for (int w = 0; w < outputMatSize.w; w++) {
				gradient[c][h][w] = 0;
			}
		}
	}

	for (int c = 0; c < poolOutputMatSize.c; c++) {//最大值位置的梯度与池化层梯度一致
		for (int h = 0; h < poolOutputMatSize.h; h++) {
			for (int w = 0; w < poolOutputMatSize.w; w++) {
				gradient[loc[c][h][w].deep][loc[c][h][w].row][loc[c][h][w].column] = poolGradient[c][h][w];
			}
		}
	}
}


/*
卷积层（后面不接池化层）局部梯度(也可以用于后面接卷积层的池化层)
需要该层的局部梯度gradient，该层的输出尺寸outputMatSize，该层的扩充圈数padding，后一层的卷积层的局部梯度nextGradient,
后一层激活前的输出nextV，后一层的输出尺寸nextOutputMatSize，后一层的卷积核权值nextKernalWeight，
后一层的卷积核尺寸nextKernalSize,后一层输出通道数nextChannels，后一层的卷积步长nextStride
*/
void convLocalGradient(float*** gradient, nSize outputMatSize, int padding, float*** nextGradient, float*** nextV, nSize nextOutputMatSize, float**** nextKernalWeight, nSize nextKernalSize, int nextChannels, int nextStride) {
	//扩充该层的梯度矩阵，扩充圈数为这层的padding
	float*** expandGradient = matEdgeExpand(gradient, outputMatSize, 2 * padding, 2 * padding);
	nSize expandGradientSize = { outputMatSize.h + 2 * padding ,outputMatSize.w + 2 * padding, outputMatSize.c };

	//扩充后面一层的梯度矩阵，扩充圈数为后一层的卷积核的尺寸-1
	float*** expanndNextGradient = matEdgeExpand(nextGradient, nextOutputMatSize, 2 * (nextKernalSize.h - 1), 2 * (nextKernalSize.w - 1));
	nSize expanndNextGradientSize = { nextOutputMatSize.h + 2 * (nextKernalSize.h - 1) ,nextOutputMatSize.w + 2 * (nextKernalSize.w - 1), nextOutputMatSize.c };

	//180°旋转后一层的卷积核
	float**** rotateKernal = matRorate180(nextKernalWeight, nextKernalSize, nextChannels);
	nSize rotateKernalSize = { nextKernalSize.h,nextKernalSize.w,nextKernalSize.c };

	//计算扩充梯度矩阵的局部梯度
	for (int c = 0; c < expandGradientSize.c; c++) {
		for (int h = 0; h < expandGradientSize.h; h++) {
			for (int w = 0; w < expandGradientSize.w; w++) {
				float sum = 0;
				for (int i = 0; i < nextChannels; i++) {
					for (int j = 0; j < rotateKernalSize.h; j++) {
						for (int k = 0; k < rotateKernalSize.w; k++) {
							int outSizeH = nextOutputMatSize.h + 2 * (nextKernalSize.h - 1);
							int outSizeW = nextOutputMatSize.w + 2 * (nextKernalSize.w - 1);
							int upperLine = ((2 * (nextKernalSize.h - 1)) + ((2 * (nextKernalSize.h - 1)) % 2)) / 2;
							int lowLine = (2 * (nextKernalSize.h - 1)) - upperLine;
							int leftColumn = ((2 * (nextKernalSize.w - 1)) + ((2 * (nextKernalSize.w - 1)) % 2)) / 2;
							int rightColumn = (2 * (nextKernalSize.w - 1)) - leftColumn;
							if ((h * nextStride + j) >= upperLine && (h * nextStride + j) <= (outSizeH - lowLine - 1) && (w * nextStride + k) >= leftColumn && (w * nextStride + k) <= (outSizeW - rightColumn - 1)) {
								sum += rotateKernal[i][c][j][k] * expanndNextGradient[i][h * nextStride + j][w * nextStride + k] * (nextV[i][h * nextStride + j - upperLine][w * nextStride + k - leftColumn] > 0 ? 1 : 0);
							}
						}
					}
				}
				expandGradient[c][h][w] = sum;
			}
		}
	}

	//取出扩充梯度矩阵中的属于本层输出的局部梯度
	int upperLine = (2 * padding + ((2 * padding) % 2)) / 2;
	int leftColumn = (2 * padding + ((2 * padding) % 2)) / 2;
	for (int c = 0; c < outputMatSize.c; c++) {
		for (int h = 0; h < outputMatSize.h; h++) {
			for (int w = 0; w < outputMatSize.w; w++) {
				gradient[c][h][w] = expandGradient[c][h + upperLine][w + leftColumn];
			}
		}
	}

	freeMatrix(expandGradient, expandGradientSize);
	freeMatrix(expanndNextGradient, expanndNextGradientSize);
	freeMatrix(rotateKernal, rotateKernalSize, nextChannels);
}


//卷积层权值更新(需要这层的卷积核，卷积核尺寸，输出通道数，这层输出的局部梯度,局部梯度尺寸（即这一层的输出尺寸），这一层的输出v(用于求导relu激活)，上一层的输出值矩阵y以及尺寸(即这一层的输入尺寸)和这一层卷积层的扩充圈数，卷积步长，学习率)
void updateConvWeight(float**** kernalWeight, nSize kernalSize, int outChannels, float*** gradient, nSize outputMatSize, float*** v, float*** lastY, nSize inputMatSize, int padding, int stride, float learningRate) {
	//先扩充上一层的输出矩阵
	float*** expandMatrix = matEdgeExpand(lastY, inputMatSize, 2 * padding, 2 * padding);
	nSize expandMatrixSize = { inputMatSize.h + 2 * padding ,inputMatSize.w + 2 * padding,inputMatSize.c };

	//更新这一层卷积核权值
	for (int n = 0; n < outChannels; n++) {
		for (int c = 0; c < kernalSize.c; c++) {
			for (int h = 0; h < kernalSize.h; h++) {
				for (int w = 0; w < kernalSize.w; w++) {
					//先计算全局梯度，再更新权重
					float Globalgradient = 0;
					for (int i = 0; i < outputMatSize.h; i++) {
						for (int j = 0; j < outputMatSize.w; j++) {
							Globalgradient += gradient[n][i][j] * (v[n][i][j] > 0 ? 1 : 0) * expandMatrix[c][h + i * stride][w + j * stride];
						}
					}
					kernalWeight[n][c][h][w] -= learningRate * Globalgradient;
				}
			}
		}
	}

	freeMatrix(expandMatrix, expandMatrixSize);
}


//全连接层权值更新(需要这一层的权值矩阵，权值矩阵尺寸，这一层的局部梯度gradient,这一层的输出v，上一层的输出y，学习率)
void updateFcWeight(float** weight, nSize weightSize, float* gradient, float* v, float* lastY, float learningRate) {
	for (int h = 0; h < weightSize.h; h++) {
		for (int w = 0; w < weightSize.w; w++) {
			weight[h][w] -= learningRate * gradient[h] * (v[h] > 0 ? 1 : 0) * lastY[w];
		}
	}
}