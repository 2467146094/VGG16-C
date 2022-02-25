#include<iostream>
#include<string>
#include<fstream>
#include"mat.h"
#include"H5.h"

#define HDF5file "module_weight/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

using namespace std;

//初始化一个卷积层
ConvLayer* initConvLayer(nSize inputMatSize, nSize kernalSize, int outChannels, int stride, int padding, const char* HDF5filename, const char* weightDatasetName, const char* biasDatasetName) {
	ConvLayer* convlayer = (ConvLayer*)malloc(sizeof(ConvLayer));

	convlayer->inputMatSize = inputMatSize;
	convlayer->kernalSize = kernalSize;
	convlayer->outChannels = outChannels;
	convlayer->stride = stride;
	convlayer->padding = padding;

	int outSizeC = outChannels;
	int outSizeH = (inputMatSize.h + 2 * padding - kernalSize.h) / stride + 1;
	int outSizeW = (inputMatSize.w + 2 * padding - kernalSize.w) / stride + 1;
	convlayer->outputMatSize.c = outSizeC;
	convlayer->outputMatSize.h = outSizeH;
	convlayer->outputMatSize.w = outSizeW;

	//初始化卷积核权值
	convlayer->kernalWeight = (float****)malloc(outChannels * sizeof(float***));
	for (int n = 0; n < outChannels; n++) {
		convlayer->kernalWeight[n] = (float***)malloc(kernalSize.c * sizeof(float**));
		for (int c = 0; c < kernalSize.c; c++) {
			convlayer->kernalWeight[n][c] = (float**)malloc(kernalSize.h * sizeof(float*));
			for (int h = 0; h < kernalSize.h; h++) {
				convlayer->kernalWeight[n][c][h] = (float*)malloc(kernalSize.w * sizeof(float));
			}
		}
	}
	readWeight(HDF5filename, weightDatasetName, convlayer->kernalSize, convlayer->outChannels, convlayer->kernalWeight);


	//初始化偏置bias；
	convlayer->bias = (float*)malloc(outChannels * sizeof(float));
	readWeight(HDF5filename, biasDatasetName, convlayer->bias);


	//分配内存空间给，进入激活函数的输入值v,经过激活函数后的输出y
	convlayer->v = (float***)malloc(outSizeC * sizeof(float**));
	convlayer->y = (float***)malloc(outSizeC * sizeof(float**));
	for (int n = 0; n < outSizeC; n++) {
		convlayer->v[n] = (float**)malloc(outSizeH * sizeof(float*));
		convlayer->y[n] = (float**)malloc(outSizeH * sizeof(float*));
		for (int h = 0; h < outSizeH; h++) {
			convlayer->v[n][h] = (float*)malloc(outSizeW * sizeof(float));
			convlayer->y[n][h] = (float*)malloc(outSizeW * sizeof(float));
		}
	}

	//分配内存空间给局部梯度gradient
	convlayer->gradient = (float***)malloc(outSizeC * sizeof(float**));
	for (int c = 0; c < outSizeC; c++) {
		convlayer->gradient[c] = (float**)malloc(outSizeH * sizeof(float*));
		for (int h = 0; h < outSizeH; h++) {
			convlayer->gradient[c][h] = (float*)malloc(outSizeW * sizeof(float));
		}
	}


	return convlayer;
}


//初始化一个池化层,poolType=0表示最大池化，1表示平均池化
PoolLayer* initPoolLayer(nSize inputMatSize, nSize poolSize, int poolType, int stride, int padding) {
	PoolLayer* poollayer = (PoolLayer*)malloc(sizeof(PoolLayer));

	poollayer->inputMatSize = inputMatSize;
	poollayer->poolSize = poolSize;
	poollayer->poolType = poolType;
	poollayer->stride = stride;
	poollayer->padding = padding;

	int outSizeC = inputMatSize.c;
	int outSizeH = (inputMatSize.h + 2 * padding - poolSize.h) / stride + 1;
	int outSizeW = (inputMatSize.w + 2 * padding - poolSize.w) / stride + 1;
	poollayer->outputMatSize.c = outSizeC;
	poollayer->outputMatSize.h = outSizeH;
	poollayer->outputMatSize.w = outSizeW;

	//分配池化后输出矩阵的空间
	poollayer->y = (float***)malloc(outSizeC * sizeof(float**));
	for (int c = 0; c < outSizeC; c++) {
		poollayer->y[c] = (float**)malloc(outSizeH * sizeof(float*));
		for (int h = 0; h < outSizeH; h++) {
			poollayer->y[c][h] = (float*)malloc(outSizeW * sizeof(float));
		}
	}

	//分配内存空间给局部梯度gradient
	poollayer->gradient = (float***)malloc(outSizeC * sizeof(float**));
	for (int c = 0; c < outSizeC; c++) {
		poollayer->gradient[c] = (float**)malloc(outSizeH * sizeof(float*));
		for (int h = 0; h < outSizeH; h++) {
			poollayer->gradient[c][h] = (float*)malloc(outSizeW * sizeof(float));
		}
	}

	//分配池化选取的值的位置loc的空间
	poollayer->loc = (valueLocation***)malloc(outSizeC * sizeof(valueLocation**));
	for (int c = 0; c < outSizeC; c++) {
		poollayer->loc[c] = (valueLocation**)malloc(outSizeH * sizeof(valueLocation*));
		for (int h = 0; h < outSizeH; h++) {
			poollayer->loc[c][h] = (valueLocation*)malloc(outSizeW * sizeof(valueLocation));
		}
	}

	return poollayer;
}


//初始化一个全连接层
FCLayer* initFCLayer(int inputNum, int outputNum, const char* HDF5filename, const char* weightDatasetName, const char* biasDatasetName) {
	FCLayer* fclayer = (FCLayer*)malloc(sizeof(FCLayer));

	fclayer->inputNum = inputNum;
	fclayer->outputNum = outputNum;
	fclayer->weightSize = { outputNum,inputNum,0 };

	//初始化权重矩阵
	fclayer->weight = (float**)malloc(outputNum * sizeof(float*));
	for (int h = 0; h < outputNum; h++) {
		fclayer->weight[h] = (float*)malloc(inputNum * sizeof(float));
	}

	nSize tempWeightSize = { inputNum,outputNum,0 };
	float** tempWeight = new float* [inputNum];
	for (int h = 0; h < inputNum; h++) {
		tempWeight[h] = new float[outputNum];
	}
	readWeight(HDF5filename, weightDatasetName, tempWeightSize, tempWeight);

	fclayer->weight = matrixTranspose(tempWeight, tempWeightSize);

	for (int h = 0; h < inputNum; h++) {
		delete[] tempWeight[h];
	}
	delete[] tempWeight;


	//初始化偏置
	fclayer->bias = new float[outputNum];
	readWeight(HDF5filename, biasDatasetName, fclayer->bias);

	//初始化 进入激活函数的输入值v, 经过激活函数后的输出y，局部梯度d
	fclayer->v = (float*)malloc(outputNum * sizeof(float));
	fclayer->y = (float*)malloc(outputNum * sizeof(float));
	fclayer->gradient = (float*)malloc(outputNum * sizeof(float));

	return  fclayer;
}




//初始化一个VGG16模型
VGG16* initVGG16() {
	VGG16* vgg16 = (VGG16*)malloc(sizeof(VGG16));

	//block1
	//初始化第1层卷积层conv1
	nSize conv1InputMatSize = { 224,224,3 };
	nSize conv1KernalSize = { 3,3,3 };
	int conv1OutChannels = 64;
	int conv1Stride = 1;
	int conv1Padding = 1;
	const char* conv1WeightDatasetName = "/block1_conv1/block1_conv1_W_1:0";
	const char* conv1BiasDatasetName = "/block1_conv1/block1_conv1_b_1:0";
	vgg16->conv1 = initConvLayer(conv1InputMatSize, conv1KernalSize, conv1OutChannels, conv1Stride, conv1Padding, HDF5file, conv1WeightDatasetName, conv1BiasDatasetName);

	//初始化第2层卷积层conv2
	nSize conv2InputMatSize = { 224,224,64 };
	nSize conv2KernalSize = { 3,3,64 };
	int conv2OutChannels = 64;
	int conv2Stride = 1;
	int conv2Padding = 1;
	const char* conv2WeightDatasetName = "/block1_conv2/block1_conv2_W_1:0";
	const char* conv2BiasDatasetName = "/block1_conv2/block1_conv2_b_1:0";
	vgg16->conv2 = initConvLayer(conv2InputMatSize, conv2KernalSize, conv2OutChannels, conv2Stride, conv2Padding, HDF5file, conv2WeightDatasetName, conv2BiasDatasetName);

	//初始化第3层池化层pool1
	nSize pool1InputMatSize = { 224,224,64 };
	nSize pool1PoolSize = { 2,2,1 };
	int pool1PoolType = 0;
	int pool1Stride = 2;
	int pool1Padding = 0;
	vgg16->pool1 = initPoolLayer(pool1InputMatSize, pool1PoolSize, pool1PoolType, pool1Stride, pool1Padding);



	//block2
	//初始化第4层卷积层conv3
	nSize conv3InputMatSize = { 112,112,64 };
	nSize conv3KernalSize = { 3,3,64 };
	int conv3OutChannels = 128;
	int conv3Stride = 1;
	int conv3Padding = 1;
	const char* conv3WeightDatasetName = "/block2_conv1/block2_conv1_W_1:0";
	const char* conv3BiasDatasetName = "/block2_conv1/block2_conv1_b_1:0";
	vgg16->conv3 = initConvLayer(conv3InputMatSize, conv3KernalSize, conv3OutChannels, conv3Stride, conv3Padding, HDF5file, conv3WeightDatasetName, conv3BiasDatasetName);

	//初始化第5层卷积层conv4
	nSize conv4InputMatSize = { 112,112,128 };
	nSize conv4KernalSize = { 3,3,128 };
	int conv4OutChannels = 128;
	int conv4Stride = 1;
	int conv4Padding = 1;
	const char* conv4WeightDatasetName = "/block2_conv2/block2_conv2_W_1:0";
	const char* conv4BiasDatasetName = "/block2_conv2/block2_conv2_b_1:0";
	vgg16->conv4 = initConvLayer(conv4InputMatSize, conv4KernalSize, conv4OutChannels, conv4Stride, conv4Padding, HDF5file, conv4WeightDatasetName, conv4BiasDatasetName);

	//初始化第6层池化层pool2
	nSize pool2InputMatSize = { 112,112,128 };
	nSize pool2PoolSize = { 2,2,1 };
	int pool2PoolType = 0;
	int pool2Stride = 2;
	int pool2Padding = 0;
	vgg16->pool2 = initPoolLayer(pool2InputMatSize, pool2PoolSize, pool2PoolType, pool2Stride, pool2Padding);



	//block3
	//初始化第7层卷积层conv5
	nSize conv5InputMatSize = { 56,56,128 };
	nSize conv5KernalSize = { 3,3,128 };
	int conv5OutChannels = 256;
	int conv5Stride = 1;
	int conv5Padding = 1;
	const char* conv5WeightDatasetName = "/block3_conv1/block3_conv1_W_1:0";
	const char* conv5BiasDatasetName = "/block3_conv1/block3_conv1_b_1:0";
	vgg16->conv5 = initConvLayer(conv5InputMatSize, conv5KernalSize, conv5OutChannels, conv5Stride, conv5Padding, HDF5file, conv5WeightDatasetName, conv5BiasDatasetName);

	//初始化第8层卷积层conv6
	nSize conv6InputMatSize = { 56,56,256 };
	nSize conv6KernalSize = { 3,3,256 };
	int conv6OutChannels = 256;
	int conv6Stride = 1;
	int conv6Padding = 1;
	const char* conv6WeightDatasetName = "/block3_conv2/block3_conv2_W_1:0";
	const char* conv6BiasDatasetName = "/block3_conv2/block3_conv2_b_1:0";
	vgg16->conv6 = initConvLayer(conv6InputMatSize, conv6KernalSize, conv6OutChannels, conv6Stride, conv6Padding, HDF5file, conv6WeightDatasetName, conv6BiasDatasetName);

	//初始化第9层卷积层conv7
	nSize conv7InputMatSize = { 56,56,256 };
	nSize conv7KernalSize = { 3,3,256 };
	int conv7OutChannels = 256;
	int conv7Stride = 1;
	int conv7Padding = 1;
	const char* conv7WeightDatasetName = "/block3_conv3/block3_conv3_W_1:0";
	const char* conv7BiasDatasetName = "/block3_conv3/block3_conv3_b_1:0";
	vgg16->conv7 = initConvLayer(conv7InputMatSize, conv7KernalSize, conv7OutChannels, conv7Stride, conv7Padding, HDF5file, conv7WeightDatasetName, conv7BiasDatasetName);

	//初始化第10层池化层pool3
	nSize pool3InputMatSize = { 56,56,256 };
	nSize pool3PoolSize = { 2,2,1 };
	int pool3PoolType = 0;
	int pool3Stride = 2;
	int pool3Padding = 0;
	vgg16->pool3 = initPoolLayer(pool3InputMatSize, pool3PoolSize, pool3PoolType, pool3Stride, pool3Padding);



	//block4
	//初始化第11层卷积层conv8
	nSize conv8InputMatSize = { 28,28,256 };
	nSize conv8KernalSize = { 3,3,256 };
	int conv8OutChannels = 512;
	int conv8Stride = 1;
	int conv8Padding = 1;
	const char* conv8WeightDatasetName = "/block4_conv1/block4_conv1_W_1:0";
	const char* conv8BiasDatasetName = "/block4_conv1/block4_conv1_b_1:0";
	vgg16->conv8 = initConvLayer(conv8InputMatSize, conv8KernalSize, conv8OutChannels, conv8Stride, conv8Padding, HDF5file, conv8WeightDatasetName, conv8BiasDatasetName);

	//初始化第12层卷积层conv9
	nSize conv9InputMatSize = { 28,28,512 };
	nSize conv9KernalSize = { 3,3,512 };
	int conv9OutChannels = 512;
	int conv9Stride = 1;
	int conv9Padding = 1;
	const char* conv9WeightDatasetName = "/block4_conv2/block4_conv2_W_1:0";
	const char* conv9BiasDatasetName = "/block4_conv2/block4_conv2_b_1:0";
	vgg16->conv9 = initConvLayer(conv9InputMatSize, conv9KernalSize, conv9OutChannels, conv9Stride, conv9Padding, HDF5file, conv9WeightDatasetName, conv9BiasDatasetName);

	//初始化第13层卷积层conv10
	nSize conv10InputMatSize = { 28,28,512 };
	nSize conv10KernalSize = { 3,3,512 };
	int conv10OutChannels = 512;
	int conv10Stride = 1;
	int conv10Padding = 1;
	const char* conv10WeightDatasetName = "/block4_conv3/block4_conv3_W_1:0";
	const char* conv10BiasDatasetName = "/block4_conv3/block4_conv3_b_1:0";
	vgg16->conv10 = initConvLayer(conv10InputMatSize, conv10KernalSize, conv10OutChannels, conv10Stride, conv10Padding, HDF5file, conv10WeightDatasetName, conv10BiasDatasetName);

	//初始化第14层池化层pool4
	nSize pool4InputMatSize = { 28,28,512 };
	nSize pool4PoolSize = { 2,2,1 };
	int pool4PoolType = 0;
	int pool4Stride = 2;
	int pool4Padding = 0;
	vgg16->pool4 = initPoolLayer(pool4InputMatSize, pool4PoolSize, pool4PoolType, pool4Stride, pool4Padding);



	//block5
	//初始化第15层卷积层conv11
	nSize conv11InputMatSize = { 14,14,512 };
	nSize conv11KernalSize = { 3,3,512 };
	int conv11OutChannels = 512;
	int conv11Stride = 1;
	int conv11Padding = 1;
	const char* conv11WeightDatasetName = "/block5_conv1/block5_conv1_W_1:0";
	const char* conv11BiasDatasetName = "/block5_conv1/block5_conv1_b_1:0";
	vgg16->conv11 = initConvLayer(conv11InputMatSize, conv11KernalSize, conv11OutChannels, conv11Stride, conv11Padding, HDF5file, conv11WeightDatasetName, conv11BiasDatasetName);

	//初始化第16层卷积层conv12
	nSize conv12InputMatSize = { 14,14,512 };
	nSize conv12KernalSize = { 3,3,512 };
	int conv12OutChannels = 512;
	int conv12Stride = 1;
	int conv12Padding = 1;
	const char* conv12WeightDatasetName = "/block5_conv2/block5_conv2_W_1:0";
	const char* conv12BiasDatasetName = "/block5_conv2/block5_conv2_b_1:0";
	vgg16->conv12 = initConvLayer(conv12InputMatSize, conv12KernalSize, conv12OutChannels, conv12Stride, conv12Padding, HDF5file, conv12WeightDatasetName, conv12BiasDatasetName);

	//初始化第17层卷积层conv13
	nSize conv13InputMatSize = { 14,14,512 };
	nSize conv13KernalSize = { 3,3,512 };
	int conv13OutChannels = 512;
	int conv13Stride = 1;
	int conv13Padding = 1;
	const char* conv13WeightDatasetName = "/block5_conv3/block5_conv3_W_1:0";
	const char* conv13BiasDatasetName = "/block5_conv3/block5_conv3_b_1:0";
	vgg16->conv13 = initConvLayer(conv13InputMatSize, conv13KernalSize, conv13OutChannels, conv13Stride, conv13Padding, HDF5file, conv13WeightDatasetName, conv13BiasDatasetName);

	//初始化第18层池化层pool5
	nSize pool5InputMatSize = { 14,14,512 };
	nSize pool5PoolSize = { 2,2,1 };
	int pool5PoolType = 0;
	int pool5Stride = 2;
	int pool5Padding = 0;
	vgg16->pool5 = initPoolLayer(pool5InputMatSize, pool5PoolSize, pool5PoolType, pool5Stride, pool5Padding);




	//初始化全连接层fc1
	int fc1inputNum = 25088;
	int fc1outputNum = 4096;
	const char* fc1WeightDatasetName = "/fc1/fc1_W_1:0";
	const char* fc1BiasDatasetName = "/fc1/fc1_b_1:0";
	vgg16->fc1 = initFCLayer(fc1inputNum, fc1outputNum, HDF5file, fc1WeightDatasetName, fc1BiasDatasetName);

	//初始化全连接层fc2
	int fc2inputNum = 4096;
	int fc2outputNum = 4096;
	const char* fc2WeightDatasetName = "/fc2/fc2_W_1:0";
	const char* fc2BiasDatasetName = "/fc2/fc2_b_1:0";
	vgg16->fc2 = initFCLayer(fc2inputNum, fc2outputNum, HDF5file, fc2WeightDatasetName, fc2BiasDatasetName);

	//初始化全连接层fc3
	int fc3inputNum = 4096;
	int fc3outputNum = 1000;
	const char* fc3WeightDatasetName = "/predictions/predictions_W_1:0";
	const char* fc3BiasDatasetName = "/predictions/predictions_b_1:0";
	vgg16->fc3 = initFCLayer(fc3inputNum, fc3outputNum, HDF5file, fc3WeightDatasetName, fc3BiasDatasetName);

	return vgg16;
}



//调用VGG16模型
string inferenceVGG16(VGG16* vgg16, float*** mat, nSize matSize) {
	free(vgg16->fc3->gradient);
	free(vgg16->fc2->gradient);
	free(vgg16->fc1->gradient);

	freeMatrix(vgg16->conv13->gradient, vgg16->conv13->outputMatSize);
	freeMatrix(vgg16->conv12->gradient, vgg16->conv12->outputMatSize);
	freeMatrix(vgg16->conv11->gradient, vgg16->conv11->outputMatSize);
	freeMatrix(vgg16->pool5->gradient, vgg16->pool5->outputMatSize);

	freeMatrix(vgg16->conv10->gradient, vgg16->conv10->outputMatSize);
	freeMatrix(vgg16->conv9->gradient, vgg16->conv9->outputMatSize);
	freeMatrix(vgg16->conv8->gradient, vgg16->conv8->outputMatSize);
	freeMatrix(vgg16->pool4->gradient, vgg16->pool4->outputMatSize);

	freeMatrix(vgg16->conv7->gradient, vgg16->conv7->outputMatSize);
	freeMatrix(vgg16->conv6->gradient, vgg16->conv6->outputMatSize);
	freeMatrix(vgg16->conv5->gradient, vgg16->conv5->outputMatSize);
	freeMatrix(vgg16->pool3->gradient, vgg16->pool3->outputMatSize);

	freeMatrix(vgg16->conv4->gradient, vgg16->conv4->outputMatSize);
	freeMatrix(vgg16->conv3->gradient, vgg16->conv3->outputMatSize);
	freeMatrix(vgg16->pool2->gradient, vgg16->pool2->outputMatSize);

	freeMatrix(vgg16->conv2->gradient, vgg16->conv2->outputMatSize);
	freeMatrix(vgg16->conv1->gradient, vgg16->conv1->outputMatSize);
	freeMatrix(vgg16->pool1->gradient, vgg16->pool1->outputMatSize);

	//进入conv1进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv1->v = conv(mat, vgg16->conv1->kernalWeight, matSize, vgg16->conv1->kernalSize, vgg16->conv1->outChannels, vgg16->conv1->stride, vgg16->conv1->padding, vgg16->conv1->bias);
	freeMatrix(mat, matSize);
	vgg16->conv1->y = relu(vgg16->conv1->v, vgg16->conv1->outputMatSize);
	freeMatrix(vgg16->conv1->v, vgg16->conv1->outputMatSize);

	//进入conv2进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv2->v = conv(vgg16->conv1->y, vgg16->conv2->kernalWeight, vgg16->conv2->inputMatSize, vgg16->conv2->kernalSize, vgg16->conv2->outChannels, vgg16->conv2->stride, vgg16->conv2->padding, vgg16->conv2->bias);
	freeMatrix(vgg16->conv1->y, vgg16->conv1->outputMatSize);
	vgg16->conv2->y = relu(vgg16->conv2->v, vgg16->conv2->outputMatSize);
	freeMatrix(vgg16->conv2->v, vgg16->conv2->outputMatSize);

	//进入pool1进行池化运算,把池化结果放入y
	vgg16->pool1->y = maxPooling(vgg16->conv2->y, vgg16->pool1->inputMatSize, vgg16->pool1->poolSize, vgg16->pool1->stride, vgg16->pool1->padding, vgg16->pool1->loc);
	freeMatrix(vgg16->conv2->y, vgg16->conv2->outputMatSize);
	freeMatrix(vgg16->pool1->loc, vgg16->pool1->outputMatSize);



	//进入conv3进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv3->v = conv(vgg16->pool1->y, vgg16->conv3->kernalWeight, vgg16->conv3->inputMatSize, vgg16->conv3->kernalSize, vgg16->conv3->outChannels, vgg16->conv3->stride, vgg16->conv3->padding, vgg16->conv3->bias);
	freeMatrix(vgg16->pool1->y, vgg16->pool1->outputMatSize);
	vgg16->conv3->y = relu(vgg16->conv3->v, vgg16->conv3->outputMatSize);
	freeMatrix(vgg16->conv3->v, vgg16->conv3->outputMatSize);

	//进入conv4进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv4->v = conv(vgg16->conv3->y, vgg16->conv4->kernalWeight, vgg16->conv4->inputMatSize, vgg16->conv4->kernalSize, vgg16->conv4->outChannels, vgg16->conv4->stride, vgg16->conv4->padding, vgg16->conv4->bias);
	freeMatrix(vgg16->conv3->y, vgg16->conv3->outputMatSize);
	vgg16->conv4->y = relu(vgg16->conv4->v, vgg16->conv4->outputMatSize);
	freeMatrix(vgg16->conv4->v, vgg16->conv4->outputMatSize);

	//进入pool2进行池化运算,把池化结果放入y
	vgg16->pool2->y = maxPooling(vgg16->conv4->y, vgg16->pool2->inputMatSize, vgg16->pool2->poolSize, vgg16->pool2->stride, vgg16->pool2->padding, vgg16->pool2->loc);
	freeMatrix(vgg16->conv4->y, vgg16->conv4->outputMatSize);
	freeMatrix(vgg16->pool2->loc, vgg16->pool2->outputMatSize);



	//进入conv5进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv5->v = conv(vgg16->pool2->y, vgg16->conv5->kernalWeight, vgg16->conv5->inputMatSize, vgg16->conv5->kernalSize, vgg16->conv5->outChannels, vgg16->conv5->stride, vgg16->conv5->padding, vgg16->conv5->bias);
	freeMatrix(vgg16->pool2->y, vgg16->pool2->outputMatSize);
	vgg16->conv5->y = relu(vgg16->conv5->v, vgg16->conv5->outputMatSize);
	freeMatrix(vgg16->conv5->v, vgg16->conv5->outputMatSize);

	//进入conv6进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv6->v = conv(vgg16->conv5->y, vgg16->conv6->kernalWeight, vgg16->conv6->inputMatSize, vgg16->conv6->kernalSize, vgg16->conv6->outChannels, vgg16->conv6->stride, vgg16->conv6->padding, vgg16->conv6->bias);
	freeMatrix(vgg16->conv5->y, vgg16->conv5->outputMatSize);
	vgg16->conv6->y = relu(vgg16->conv6->v, vgg16->conv6->outputMatSize);
	freeMatrix(vgg16->conv6->v, vgg16->conv6->outputMatSize);

	//进入conv7进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv7->v = conv(vgg16->conv6->y, vgg16->conv7->kernalWeight, vgg16->conv7->inputMatSize, vgg16->conv7->kernalSize, vgg16->conv7->outChannels, vgg16->conv7->stride, vgg16->conv7->padding, vgg16->conv7->bias);
	freeMatrix(vgg16->conv6->y, vgg16->conv6->outputMatSize);
	vgg16->conv7->y = relu(vgg16->conv7->v, vgg16->conv7->outputMatSize);
	freeMatrix(vgg16->conv7->v, vgg16->conv7->outputMatSize);

	//进入pool3进行池化运算,把池化结果放入y
	vgg16->pool3->y = maxPooling(vgg16->conv7->y, vgg16->pool3->inputMatSize, vgg16->pool3->poolSize, vgg16->pool3->stride, vgg16->pool3->padding, vgg16->pool3->loc);
	freeMatrix(vgg16->conv7->y, vgg16->conv7->outputMatSize);
	freeMatrix(vgg16->pool3->loc, vgg16->pool3->outputMatSize);



	//进入conv8进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv8->v = conv(vgg16->pool3->y, vgg16->conv8->kernalWeight, vgg16->conv8->inputMatSize, vgg16->conv8->kernalSize, vgg16->conv8->outChannels, vgg16->conv8->stride, vgg16->conv8->padding, vgg16->conv8->bias);
	freeMatrix(vgg16->pool3->y, vgg16->pool3->outputMatSize);
	vgg16->conv8->y = relu(vgg16->conv8->v, vgg16->conv8->outputMatSize);
	freeMatrix(vgg16->conv8->v, vgg16->conv8->outputMatSize);

	//进入conv9进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv9->v = conv(vgg16->conv8->y, vgg16->conv9->kernalWeight, vgg16->conv9->inputMatSize, vgg16->conv9->kernalSize, vgg16->conv9->outChannels, vgg16->conv9->stride, vgg16->conv9->padding, vgg16->conv9->bias);
	freeMatrix(vgg16->conv8->y, vgg16->conv8->outputMatSize);
	vgg16->conv9->y = relu(vgg16->conv9->v, vgg16->conv9->outputMatSize);
	freeMatrix(vgg16->conv9->v, vgg16->conv9->outputMatSize);

	//进入conv10进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv10->v = conv(vgg16->conv9->y, vgg16->conv10->kernalWeight, vgg16->conv10->inputMatSize, vgg16->conv10->kernalSize, vgg16->conv10->outChannels, vgg16->conv10->stride, vgg16->conv10->padding, vgg16->conv10->bias);
	freeMatrix(vgg16->conv9->y, vgg16->conv9->outputMatSize);
	vgg16->conv10->y = relu(vgg16->conv10->v, vgg16->conv10->outputMatSize);
	freeMatrix(vgg16->conv10->v, vgg16->conv10->outputMatSize);

	//进入pool4进行池化运算,把池化结果放入y
	vgg16->pool4->y = maxPooling(vgg16->conv10->y, vgg16->pool4->inputMatSize, vgg16->pool4->poolSize, vgg16->pool4->stride, vgg16->pool4->padding, vgg16->pool4->loc);
	freeMatrix(vgg16->conv10->y, vgg16->conv10->outputMatSize);
	freeMatrix(vgg16->pool4->loc, vgg16->pool4->outputMatSize);



	//进入conv11进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv11->v = conv(vgg16->pool4->y, vgg16->conv11->kernalWeight, vgg16->conv11->inputMatSize, vgg16->conv11->kernalSize, vgg16->conv11->outChannels, vgg16->conv11->stride, vgg16->conv11->padding, vgg16->conv11->bias);
	freeMatrix(vgg16->pool4->y, vgg16->pool4->outputMatSize);
	vgg16->conv11->y = relu(vgg16->conv11->v, vgg16->conv11->outputMatSize);
	freeMatrix(vgg16->conv11->v, vgg16->conv11->outputMatSize);

	//进入conv12进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv12->v = conv(vgg16->conv11->y, vgg16->conv12->kernalWeight, vgg16->conv12->inputMatSize, vgg16->conv12->kernalSize, vgg16->conv12->outChannels, vgg16->conv12->stride, vgg16->conv12->padding, vgg16->conv12->bias);
	freeMatrix(vgg16->conv11->y, vgg16->conv11->outputMatSize);
	vgg16->conv12->y = relu(vgg16->conv12->v, vgg16->conv12->outputMatSize);
	freeMatrix(vgg16->conv12->v, vgg16->conv12->outputMatSize);

	//进入conv13进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv13->v = conv(vgg16->conv12->y, vgg16->conv13->kernalWeight, vgg16->conv13->inputMatSize, vgg16->conv13->kernalSize, vgg16->conv13->outChannels, vgg16->conv13->stride, vgg16->conv13->padding, vgg16->conv13->bias);
	freeMatrix(vgg16->conv12->y, vgg16->conv12->outputMatSize);
	vgg16->conv13->y = relu(vgg16->conv13->v, vgg16->conv13->outputMatSize);
	freeMatrix(vgg16->conv13->v, vgg16->conv13->outputMatSize);

	//进入pool5进行池化运算,把池化结果放入y
	vgg16->pool5->y = maxPooling(vgg16->conv13->y, vgg16->pool5->inputMatSize, vgg16->pool5->poolSize, vgg16->pool5->stride, vgg16->pool5->padding, vgg16->pool5->loc);
	freeMatrix(vgg16->conv13->y, vgg16->conv13->outputMatSize);
	freeMatrix(vgg16->pool5->loc, vgg16->pool5->outputMatSize);


	//把特征展平成一维数组
	float* flat = flatten(vgg16->pool5->y, vgg16->pool5->outputMatSize);
	freeMatrix(vgg16->pool5->y, vgg16->pool5->outputMatSize);

	//进入fc1进行全连接运算,把全连接结果放入v,把relu激活结果放入y
	vgg16->fc1->v = fc(vgg16->fc1->weight, vgg16->fc1->weightSize, flat, vgg16->fc1->inputNum, vgg16->fc1->bias);
	free(flat);
	vgg16->fc1->y = relu(vgg16->fc1->v, vgg16->fc1->outputNum);
	free(vgg16->fc1->v);

	//进入fc2进行全连接运算,把全连接结果放入v,把relu激活结果放入y
	vgg16->fc2->v = fc(vgg16->fc2->weight, vgg16->fc2->weightSize, vgg16->fc1->y, vgg16->fc2->inputNum, vgg16->fc2->bias);
	free(vgg16->fc1->y);
	vgg16->fc2->y = relu(vgg16->fc2->v, vgg16->fc2->outputNum);
	free(vgg16->fc2->v);

	//进入fc3进行全连接运算,把全连接结果放入v,把relu激活结果放入y
	vgg16->fc3->v = fc(vgg16->fc3->weight, vgg16->fc3->weightSize, vgg16->fc2->y, vgg16->fc3->inputNum, vgg16->fc3->bias);
	free(vgg16->fc2->y);
	vgg16->fc3->y = relu(vgg16->fc3->v, vgg16->fc3->outputNum);
	free(vgg16->fc3->v);

	//概率输出每个种类的结果
	float* softMax = softmax(vgg16->fc3->y, vgg16->fc3->outputNum);
	free(vgg16->fc3->y);

	//返回概率最大的种类
	int max = argMax(softMax, vgg16->fc3->outputNum);

	string label;
	ifstream fin("C:/Users/JackSon/testImage/imageNet标签.txt");
	for (int i = 0; i <= max; i++) {
		getline(fin, label);
	}
	label = label.substr(12);



	return label;

}



//训练VGG16模型
void trainingVGG16(VGG16* vgg16, float*** mat, nSize matSize, int label, float learningRate, string loss) {

	//进入conv1进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv1->v = conv(mat, vgg16->conv1->kernalWeight, matSize, vgg16->conv1->kernalSize, vgg16->conv1->outChannels, vgg16->conv1->stride, vgg16->conv1->padding, vgg16->conv1->bias);
	vgg16->conv1->y = relu(vgg16->conv1->v, vgg16->conv1->outputMatSize);

	//进入conv2进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv2->v = conv(vgg16->conv1->y, vgg16->conv2->kernalWeight, vgg16->conv2->inputMatSize, vgg16->conv2->kernalSize, vgg16->conv2->outChannels, vgg16->conv2->stride, vgg16->conv2->padding, vgg16->conv2->bias);
	vgg16->conv2->y = relu(vgg16->conv2->v, vgg16->conv2->outputMatSize);

	//进入pool1进行池化运算,把池化结果放入y
	vgg16->pool1->y = maxPooling(vgg16->conv2->y, vgg16->pool1->inputMatSize, vgg16->pool1->poolSize, vgg16->pool1->stride, vgg16->pool1->padding, vgg16->pool1->loc);



	//进入conv3进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv3->v = conv(vgg16->pool1->y, vgg16->conv3->kernalWeight, vgg16->conv3->inputMatSize, vgg16->conv3->kernalSize, vgg16->conv3->outChannels, vgg16->conv3->stride, vgg16->conv3->padding, vgg16->conv3->bias);
	vgg16->conv3->y = relu(vgg16->conv3->v, vgg16->conv3->outputMatSize);

	//进入conv4进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv4->v = conv(vgg16->conv3->y, vgg16->conv4->kernalWeight, vgg16->conv4->inputMatSize, vgg16->conv4->kernalSize, vgg16->conv4->outChannels, vgg16->conv4->stride, vgg16->conv4->padding, vgg16->conv4->bias);
	vgg16->conv4->y = relu(vgg16->conv4->v, vgg16->conv4->outputMatSize);

	//进入pool2进行池化运算,把池化结果放入y
	vgg16->pool2->y = maxPooling(vgg16->conv4->y, vgg16->pool2->inputMatSize, vgg16->pool2->poolSize, vgg16->pool2->stride, vgg16->pool2->padding, vgg16->pool2->loc);



	//进入conv5进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv5->v = conv(vgg16->pool2->y, vgg16->conv5->kernalWeight, vgg16->conv5->inputMatSize, vgg16->conv5->kernalSize, vgg16->conv5->outChannels, vgg16->conv5->stride, vgg16->conv5->padding, vgg16->conv5->bias);
	vgg16->conv5->y = relu(vgg16->conv5->v, vgg16->conv5->outputMatSize);

	//进入conv6进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv6->v = conv(vgg16->conv5->y, vgg16->conv6->kernalWeight, vgg16->conv6->inputMatSize, vgg16->conv6->kernalSize, vgg16->conv6->outChannels, vgg16->conv6->stride, vgg16->conv6->padding, vgg16->conv6->bias);
	vgg16->conv6->y = relu(vgg16->conv6->v, vgg16->conv6->outputMatSize);

	//进入conv7进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv7->v = conv(vgg16->conv6->y, vgg16->conv7->kernalWeight, vgg16->conv7->inputMatSize, vgg16->conv7->kernalSize, vgg16->conv7->outChannels, vgg16->conv7->stride, vgg16->conv7->padding, vgg16->conv7->bias);
	vgg16->conv7->y = relu(vgg16->conv7->v, vgg16->conv7->outputMatSize);

	//进入pool3进行池化运算,把池化结果放入y
	vgg16->pool3->y = maxPooling(vgg16->conv7->y, vgg16->pool3->inputMatSize, vgg16->pool3->poolSize, vgg16->pool3->stride, vgg16->pool3->padding, vgg16->pool3->loc);



	//进入conv8进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv8->v = conv(vgg16->pool3->y, vgg16->conv8->kernalWeight, vgg16->conv8->inputMatSize, vgg16->conv8->kernalSize, vgg16->conv8->outChannels, vgg16->conv8->stride, vgg16->conv8->padding, vgg16->conv8->bias);
	vgg16->conv8->y = relu(vgg16->conv8->v, vgg16->conv8->outputMatSize);

	//进入conv9进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv9->v = conv(vgg16->conv8->y, vgg16->conv9->kernalWeight, vgg16->conv9->inputMatSize, vgg16->conv9->kernalSize, vgg16->conv9->outChannels, vgg16->conv9->stride, vgg16->conv9->padding, vgg16->conv9->bias);
	vgg16->conv9->y = relu(vgg16->conv9->v, vgg16->conv9->outputMatSize);

	//进入conv10进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv10->v = conv(vgg16->conv9->y, vgg16->conv10->kernalWeight, vgg16->conv10->inputMatSize, vgg16->conv10->kernalSize, vgg16->conv10->outChannels, vgg16->conv10->stride, vgg16->conv10->padding, vgg16->conv10->bias);
	vgg16->conv10->y = relu(vgg16->conv10->v, vgg16->conv10->outputMatSize);

	//进入pool4进行池化运算,把池化结果放入y
	vgg16->pool4->y = maxPooling(vgg16->conv10->y, vgg16->pool4->inputMatSize, vgg16->pool4->poolSize, vgg16->pool4->stride, vgg16->pool4->padding, vgg16->pool4->loc);



	//进入conv11进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv11->v = conv(vgg16->pool4->y, vgg16->conv11->kernalWeight, vgg16->conv11->inputMatSize, vgg16->conv11->kernalSize, vgg16->conv11->outChannels, vgg16->conv11->stride, vgg16->conv11->padding, vgg16->conv11->bias);
	vgg16->conv11->y = relu(vgg16->conv11->v, vgg16->conv11->outputMatSize);

	//进入conv12进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv12->v = conv(vgg16->conv11->y, vgg16->conv12->kernalWeight, vgg16->conv12->inputMatSize, vgg16->conv12->kernalSize, vgg16->conv12->outChannels, vgg16->conv12->stride, vgg16->conv12->padding, vgg16->conv12->bias);
	vgg16->conv12->y = relu(vgg16->conv12->v, vgg16->conv12->outputMatSize);

	//进入conv13进行卷积运算,把卷积结果放入v，把relu激活结果放入y
	vgg16->conv13->v = conv(vgg16->conv12->y, vgg16->conv13->kernalWeight, vgg16->conv13->inputMatSize, vgg16->conv13->kernalSize, vgg16->conv13->outChannels, vgg16->conv13->stride, vgg16->conv13->padding, vgg16->conv13->bias);
	vgg16->conv13->y = relu(vgg16->conv13->v, vgg16->conv13->outputMatSize);

	//进入pool5进行池化运算,把池化结果放入y
	vgg16->pool5->y = maxPooling(vgg16->conv13->y, vgg16->pool5->inputMatSize, vgg16->pool5->poolSize, vgg16->pool5->stride, vgg16->pool5->padding, vgg16->pool5->loc);



	//把特征展平成一维数组
	float* flat = flatten(vgg16->pool5->y, vgg16->pool5->outputMatSize);

	//进入fc1进行全连接运算,把全连接结果放入v，把relu激活结果放入y
	vgg16->fc1->v = fc(vgg16->fc1->weight, vgg16->fc1->weightSize, flat, vgg16->fc1->inputNum, vgg16->fc1->bias);
	vgg16->fc1->y = relu(vgg16->fc1->v, vgg16->fc1->outputNum);

	//进入fc2进行全连接运算,把全连接结果放入v，把relu激活结果放入y
	vgg16->fc2->v = fc(vgg16->fc2->weight, vgg16->fc2->weightSize, vgg16->fc1->y, vgg16->fc2->inputNum, vgg16->fc2->bias);
	vgg16->fc2->y = relu(vgg16->fc2->v, vgg16->fc2->outputNum);

	//进入fc3进行全连接运算,把全连接结果放入v，把relu激活结果放入y
	vgg16->fc3->v = fc(vgg16->fc3->weight, vgg16->fc3->weightSize, vgg16->fc2->y, vgg16->fc3->inputNum, vgg16->fc3->bias);
	vgg16->fc3->y = relu(vgg16->fc3->v, vgg16->fc3->outputNum);





	//从后往前求每层的局部梯度（损失函数为均方误差）
	//fc3的局部梯度
	for (int i = 0; i < vgg16->fc3->outputNum; i++) {
		if (i != label) {
			vgg16->fc3->gradient[i] = vgg16->fc3->y[i];
		}
		else {
			vgg16->fc3->gradient[i] = vgg16->fc3->y[i] - 1;
		}
	}

	//fc2的局部梯度
	for (int i = 0; i < vgg16->fc2->outputNum; i++) {
		vgg16->fc2->gradient[i] = 0;
		for (int j = 0; j < vgg16->fc3->outputNum; j++) {
			vgg16->fc2->gradient[i] += vgg16->fc3->gradient[j] * (vgg16->fc3->v[j] > 0 ? 1 : 0) * vgg16->fc3->weight[j][i];
		}
	}

	//fc1的局部梯度
	for (int i = 0; i < vgg16->fc1->outputNum; i++) {
		vgg16->fc1->gradient[i] = 0;
		for (int j = 0; j < vgg16->fc2->outputNum; j++) {
			vgg16->fc1->gradient[i] += vgg16->fc2->gradient[j] * (vgg16->fc2->v[j] > 0 ? 1 : 0) * vgg16->fc2->weight[j][i];
		}
	}



	//pool5的局部梯度
	for (int c = 0; c < vgg16->pool5->outputMatSize.c; c++) {
		for (int h = 0; h < vgg16->pool5->outputMatSize.h; h++) {
			for (int w = 0; w < vgg16->pool5->outputMatSize.w; w++) {
				vgg16->pool5->gradient[c][h][w] = 0;
				for (int i = 0; i < vgg16->fc1->outputNum; i++) {
					vgg16->pool5->gradient[c][h][w] += vgg16->fc1->gradient[i] * (vgg16->fc1->v[i] > 0 ? 1 : 0) * vgg16->fc1->weight[i][h * matSize.w * matSize.c + w * matSize.c + c];
				}
			}
		}
	}

	//conv13的局部梯度
	convLocalGradientBeforePooling(vgg16->conv13->gradient, vgg16->conv13->outputMatSize, vgg16->pool5->gradient, vgg16->pool5->outputMatSize, vgg16->pool5->loc);

	//conv12的局部梯度
	convLocalGradient(vgg16->conv12->gradient, vgg16->conv12->outputMatSize, vgg16->conv12->padding, vgg16->conv13->gradient, vgg16->conv13->v, vgg16->conv13->outputMatSize, vgg16->conv13->kernalWeight, vgg16->conv13->kernalSize, vgg16->conv13->outChannels, vgg16->conv13->stride);

	//conv11的局部梯度
	convLocalGradient(vgg16->conv11->gradient, vgg16->conv11->outputMatSize, vgg16->conv11->padding, vgg16->conv12->gradient, vgg16->conv12->v, vgg16->conv12->outputMatSize, vgg16->conv12->kernalWeight, vgg16->conv12->kernalSize, vgg16->conv12->outChannels, vgg16->conv12->stride);



	//pool4的局部梯度
	convLocalGradient(vgg16->pool4->gradient, vgg16->pool4->outputMatSize, vgg16->pool4->padding, vgg16->conv11->gradient, vgg16->conv11->v, vgg16->conv11->outputMatSize, vgg16->conv11->kernalWeight, vgg16->conv11->kernalSize, vgg16->conv11->outChannels, vgg16->conv11->stride);

	//conv10的局部梯度
	convLocalGradientBeforePooling(vgg16->conv10->gradient, vgg16->conv10->outputMatSize, vgg16->pool4->gradient, vgg16->pool4->outputMatSize, vgg16->pool4->loc);

	//conv9的局部梯度
	convLocalGradient(vgg16->conv9->gradient, vgg16->conv9->outputMatSize, vgg16->conv9->padding, vgg16->conv10->gradient, vgg16->conv10->v, vgg16->conv10->outputMatSize, vgg16->conv10->kernalWeight, vgg16->conv10->kernalSize, vgg16->conv10->outChannels, vgg16->conv10->stride);

	//conv8的局部梯度
	convLocalGradient(vgg16->conv8->gradient, vgg16->conv8->outputMatSize, vgg16->conv8->padding, vgg16->conv9->gradient, vgg16->conv9->v, vgg16->conv9->outputMatSize, vgg16->conv9->kernalWeight, vgg16->conv9->kernalSize, vgg16->conv9->outChannels, vgg16->conv9->stride);



	//pool3的局部梯度
	convLocalGradient(vgg16->pool3->gradient, vgg16->pool3->outputMatSize, vgg16->pool3->padding, vgg16->conv8->gradient, vgg16->conv8->v, vgg16->conv8->outputMatSize, vgg16->conv8->kernalWeight, vgg16->conv8->kernalSize, vgg16->conv8->outChannels, vgg16->conv8->stride);

	//conv7的局部梯度
	convLocalGradientBeforePooling(vgg16->conv7->gradient, vgg16->conv7->outputMatSize, vgg16->pool3->gradient, vgg16->pool3->outputMatSize, vgg16->pool3->loc);

	//conv6的局部梯度
	convLocalGradient(vgg16->conv6->gradient, vgg16->conv6->outputMatSize, vgg16->conv6->padding, vgg16->conv7->gradient, vgg16->conv7->v, vgg16->conv7->outputMatSize, vgg16->conv7->kernalWeight, vgg16->conv7->kernalSize, vgg16->conv7->outChannels, vgg16->conv7->stride);

	//conv5的局部梯度
	convLocalGradient(vgg16->conv5->gradient, vgg16->conv5->outputMatSize, vgg16->conv5->padding, vgg16->conv6->gradient, vgg16->conv6->v, vgg16->conv6->outputMatSize, vgg16->conv6->kernalWeight, vgg16->conv6->kernalSize, vgg16->conv6->outChannels, vgg16->conv6->stride);



	//pool2的局部梯度
	convLocalGradient(vgg16->pool2->gradient, vgg16->pool2->outputMatSize, vgg16->pool2->padding, vgg16->conv5->gradient, vgg16->conv5->v, vgg16->conv5->outputMatSize, vgg16->conv5->kernalWeight, vgg16->conv5->kernalSize, vgg16->conv5->outChannels, vgg16->conv5->stride);

	//conv4的局部梯度
	convLocalGradientBeforePooling(vgg16->conv4->gradient, vgg16->conv4->outputMatSize, vgg16->pool2->gradient, vgg16->pool2->outputMatSize, vgg16->pool2->loc);

	//conv3的局部梯度
	convLocalGradient(vgg16->conv3->gradient, vgg16->conv3->outputMatSize, vgg16->conv3->padding, vgg16->conv4->gradient, vgg16->conv4->v, vgg16->conv4->outputMatSize, vgg16->conv4->kernalWeight, vgg16->conv4->kernalSize, vgg16->conv4->outChannels, vgg16->conv4->stride);



	//pool1的局部梯度
	convLocalGradient(vgg16->pool1->gradient, vgg16->pool1->outputMatSize, vgg16->pool1->padding, vgg16->conv3->gradient, vgg16->conv3->v, vgg16->conv3->outputMatSize, vgg16->conv3->kernalWeight, vgg16->conv3->kernalSize, vgg16->conv3->outChannels, vgg16->conv3->stride);

	//conv2的局部梯度
	convLocalGradientBeforePooling(vgg16->conv2->gradient, vgg16->conv2->outputMatSize, vgg16->pool1->gradient, vgg16->pool1->outputMatSize, vgg16->pool1->loc);

	//conv1的局部梯度
	convLocalGradient(vgg16->conv1->gradient, vgg16->conv1->outputMatSize, vgg16->conv1->padding, vgg16->conv2->gradient, vgg16->conv2->v, vgg16->conv2->outputMatSize, vgg16->conv2->kernalWeight, vgg16->conv2->kernalSize, vgg16->conv2->outChannels, vgg16->conv2->stride);






	//从前往后权重更新
	//对conv1层进行权重更新
	updateConvWeight(vgg16->conv1->kernalWeight, vgg16->conv1->kernalSize, vgg16->conv1->outChannels, vgg16->conv1->gradient, vgg16->conv1->outputMatSize, vgg16->conv1->v, mat, vgg16->conv1->inputMatSize, vgg16->conv1->padding, vgg16->conv1->stride, learningRate);

	//对conv2层进行权重更新
	updateConvWeight(vgg16->conv2->kernalWeight, vgg16->conv2->kernalSize, vgg16->conv2->outChannels, vgg16->conv2->gradient, vgg16->conv2->outputMatSize, vgg16->conv2->v, vgg16->conv1->y, vgg16->conv2->inputMatSize, vgg16->conv2->padding, vgg16->conv2->stride, learningRate);

	//对conv3层进行权重更新
	updateConvWeight(vgg16->conv3->kernalWeight, vgg16->conv3->kernalSize, vgg16->conv3->outChannels, vgg16->conv3->gradient, vgg16->conv3->outputMatSize, vgg16->conv3->v, vgg16->pool1->y, vgg16->conv3->inputMatSize, vgg16->conv3->padding, vgg16->conv3->stride, learningRate);

	//对conv4层进行权重更新
	updateConvWeight(vgg16->conv4->kernalWeight, vgg16->conv4->kernalSize, vgg16->conv4->outChannels, vgg16->conv4->gradient, vgg16->conv4->outputMatSize, vgg16->conv4->v, vgg16->conv3->y, vgg16->conv4->inputMatSize, vgg16->conv4->padding, vgg16->conv4->stride, learningRate);

	//对conv5层进行权重更新
	updateConvWeight(vgg16->conv5->kernalWeight, vgg16->conv5->kernalSize, vgg16->conv5->outChannels, vgg16->conv5->gradient, vgg16->conv5->outputMatSize, vgg16->conv5->v, vgg16->pool2->y, vgg16->conv5->inputMatSize, vgg16->conv5->padding, vgg16->conv5->stride, learningRate);

	//对conv6层进行权重更新
	updateConvWeight(vgg16->conv6->kernalWeight, vgg16->conv6->kernalSize, vgg16->conv6->outChannels, vgg16->conv6->gradient, vgg16->conv6->outputMatSize, vgg16->conv6->v, vgg16->conv5->y, vgg16->conv6->inputMatSize, vgg16->conv6->padding, vgg16->conv6->stride, learningRate);

	//对conv7层进行权重更新
	updateConvWeight(vgg16->conv7->kernalWeight, vgg16->conv7->kernalSize, vgg16->conv7->outChannels, vgg16->conv7->gradient, vgg16->conv7->outputMatSize, vgg16->conv7->v, vgg16->conv6->y, vgg16->conv7->inputMatSize, vgg16->conv7->padding, vgg16->conv7->stride, learningRate);

	//对conv8层进行权重更新
	updateConvWeight(vgg16->conv8->kernalWeight, vgg16->conv8->kernalSize, vgg16->conv8->outChannels, vgg16->conv8->gradient, vgg16->conv8->outputMatSize, vgg16->conv8->v, vgg16->pool3->y, vgg16->conv8->inputMatSize, vgg16->conv8->padding, vgg16->conv8->stride, learningRate);

	//对conv9层进行权重更新
	updateConvWeight(vgg16->conv9->kernalWeight, vgg16->conv9->kernalSize, vgg16->conv9->outChannels, vgg16->conv9->gradient, vgg16->conv9->outputMatSize, vgg16->conv9->v, vgg16->conv8->y, vgg16->conv9->inputMatSize, vgg16->conv9->padding, vgg16->conv9->stride, learningRate);

	//对conv10层进行权重更新
	updateConvWeight(vgg16->conv10->kernalWeight, vgg16->conv10->kernalSize, vgg16->conv10->outChannels, vgg16->conv10->gradient, vgg16->conv10->outputMatSize, vgg16->conv10->v, vgg16->conv9->y, vgg16->conv10->inputMatSize, vgg16->conv10->padding, vgg16->conv10->stride, learningRate);

	//对conv11层进行权重更新
	updateConvWeight(vgg16->conv11->kernalWeight, vgg16->conv11->kernalSize, vgg16->conv11->outChannels, vgg16->conv11->gradient, vgg16->conv11->outputMatSize, vgg16->conv11->v, vgg16->pool4->y, vgg16->conv11->inputMatSize, vgg16->conv11->padding, vgg16->conv11->stride, learningRate);

	//对conv12层进行权重更新
	updateConvWeight(vgg16->conv12->kernalWeight, vgg16->conv12->kernalSize, vgg16->conv12->outChannels, vgg16->conv12->gradient, vgg16->conv12->outputMatSize, vgg16->conv12->v, vgg16->conv11->y, vgg16->conv12->inputMatSize, vgg16->conv12->padding, vgg16->conv12->stride, learningRate);

	//对conv13层进行权重更新
	updateConvWeight(vgg16->conv13->kernalWeight, vgg16->conv13->kernalSize, vgg16->conv13->outChannels, vgg16->conv13->gradient, vgg16->conv13->outputMatSize, vgg16->conv13->v, vgg16->conv12->y, vgg16->conv13->inputMatSize, vgg16->conv13->padding, vgg16->conv13->stride, learningRate);

	//对fc1层进行权重更新
	updateFcWeight(vgg16->fc1->weight, vgg16->fc1->weightSize, vgg16->fc1->gradient, vgg16->fc1->v, flat, learningRate);

	//对fc2层进行权重更新
	updateFcWeight(vgg16->fc2->weight, vgg16->fc2->weightSize, vgg16->fc2->gradient, vgg16->fc2->v, vgg16->fc1->y, learningRate);

	//对fc3层进行权重更新
	updateFcWeight(vgg16->fc3->weight, vgg16->fc3->weightSize, vgg16->fc3->gradient, vgg16->fc3->v, vgg16->fc2->y, learningRate);




	//释放所有内存
	freeMatrix(mat, matSize);

	freeMatrix(vgg16->conv1->v, vgg16->conv1->outputMatSize);
	freeMatrix(vgg16->conv1->y, vgg16->conv1->outputMatSize);
	freeMatrix(vgg16->conv1->gradient, vgg16->conv1->outputMatSize);

	freeMatrix(vgg16->conv2->v, vgg16->conv2->outputMatSize);
	freeMatrix(vgg16->conv2->y, vgg16->conv2->outputMatSize);
	freeMatrix(vgg16->conv2->gradient, vgg16->conv2->outputMatSize);

	freeMatrix(vgg16->pool1->y, vgg16->pool1->outputMatSize);
	freeMatrix(vgg16->pool1->gradient, vgg16->pool1->outputMatSize);
	freeMatrix(vgg16->pool1->loc, vgg16->pool1->outputMatSize);



	freeMatrix(vgg16->conv3->v, vgg16->conv3->outputMatSize);
	freeMatrix(vgg16->conv3->y, vgg16->conv3->outputMatSize);
	freeMatrix(vgg16->conv3->gradient, vgg16->conv3->outputMatSize);

	freeMatrix(vgg16->conv4->v, vgg16->conv4->outputMatSize);
	freeMatrix(vgg16->conv4->y, vgg16->conv4->outputMatSize);
	freeMatrix(vgg16->conv4->gradient, vgg16->conv4->outputMatSize);

	freeMatrix(vgg16->pool2->y, vgg16->pool2->outputMatSize);
	freeMatrix(vgg16->pool2->gradient, vgg16->pool2->outputMatSize);
	freeMatrix(vgg16->pool2->loc, vgg16->pool2->outputMatSize);



	freeMatrix(vgg16->conv5->v, vgg16->conv5->outputMatSize);
	freeMatrix(vgg16->conv5->y, vgg16->conv5->outputMatSize);
	freeMatrix(vgg16->conv5->gradient, vgg16->conv5->outputMatSize);

	freeMatrix(vgg16->conv6->v, vgg16->conv6->outputMatSize);
	freeMatrix(vgg16->conv6->y, vgg16->conv6->outputMatSize);
	freeMatrix(vgg16->conv6->gradient, vgg16->conv6->outputMatSize);

	freeMatrix(vgg16->conv7->v, vgg16->conv7->outputMatSize);
	freeMatrix(vgg16->conv7->y, vgg16->conv7->outputMatSize);
	freeMatrix(vgg16->conv7->gradient, vgg16->conv7->outputMatSize);

	freeMatrix(vgg16->pool3->y, vgg16->pool3->outputMatSize);
	freeMatrix(vgg16->pool3->gradient, vgg16->pool3->outputMatSize);
	freeMatrix(vgg16->pool3->loc, vgg16->pool3->outputMatSize);



	freeMatrix(vgg16->conv8->v, vgg16->conv8->outputMatSize);
	freeMatrix(vgg16->conv8->y, vgg16->conv8->outputMatSize);
	freeMatrix(vgg16->conv8->gradient, vgg16->conv8->outputMatSize);

	freeMatrix(vgg16->conv9->v, vgg16->conv9->outputMatSize);
	freeMatrix(vgg16->conv9->y, vgg16->conv9->outputMatSize);
	freeMatrix(vgg16->conv9->gradient, vgg16->conv9->outputMatSize);

	freeMatrix(vgg16->conv10->v, vgg16->conv10->outputMatSize);
	freeMatrix(vgg16->conv10->y, vgg16->conv10->outputMatSize);
	freeMatrix(vgg16->conv10->gradient, vgg16->conv10->outputMatSize);

	freeMatrix(vgg16->pool4->y, vgg16->pool4->outputMatSize);
	freeMatrix(vgg16->pool4->gradient, vgg16->pool4->outputMatSize);
	freeMatrix(vgg16->pool4->loc, vgg16->pool4->outputMatSize);



	freeMatrix(vgg16->conv11->v, vgg16->conv11->outputMatSize);
	freeMatrix(vgg16->conv11->y, vgg16->conv11->outputMatSize);
	freeMatrix(vgg16->conv11->gradient, vgg16->conv11->outputMatSize);

	freeMatrix(vgg16->conv12->v, vgg16->conv12->outputMatSize);
	freeMatrix(vgg16->conv12->y, vgg16->conv12->outputMatSize);
	freeMatrix(vgg16->conv12->gradient, vgg16->conv12->outputMatSize);

	freeMatrix(vgg16->conv13->v, vgg16->conv13->outputMatSize);
	freeMatrix(vgg16->conv13->y, vgg16->conv13->outputMatSize);
	freeMatrix(vgg16->conv13->gradient, vgg16->conv13->outputMatSize);

	freeMatrix(vgg16->pool5->y, vgg16->pool5->outputMatSize);
	freeMatrix(vgg16->pool5->gradient, vgg16->pool5->outputMatSize);
	freeMatrix(vgg16->pool5->loc, vgg16->pool5->outputMatSize);



	free(flat);

	free(vgg16->fc1->v);
	free(vgg16->fc1->y);
	free(vgg16->fc1->gradient);

	free(vgg16->fc2->v);
	free(vgg16->fc2->y);
	free(vgg16->fc2->gradient);

	free(vgg16->fc3->v);
	free(vgg16->fc3->y);
	free(vgg16->fc3->gradient);

}