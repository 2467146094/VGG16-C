#include"H5.h"
#include"hdf5.h"
#include"cnn.h"

//读取一维数据：偏置
void readWeight(const char* HDF5filename, const char* datasetName, float* bias) {

	hid_t file_id;
	herr_t status;
	hid_t data_id;

	//打开h5文件
	file_id = H5Fopen(HDF5filename, H5F_ACC_RDONLY, H5P_DEFAULT);

	//打开数据集
	data_id = H5Dopen(file_id, datasetName, H5P_DEFAULT);

	//读取数据到data
	status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, bias);

	//关闭数据文件和h5文件
	status = H5Dclose(data_id);
	status = H5Fclose(file_id);
}



//读取四维数据：卷积核权值
void readWeight(const char* HDF5filename, const char* datasetName, nSize kernalSize, int outChannels, float**** kernalWeight) {

	float* data = new float[kernalSize.h * kernalSize.w * kernalSize.c * outChannels];

	hid_t file_id;
	herr_t status;
	hid_t data_id;

	//打开h5文件
	file_id = H5Fopen(HDF5filename, H5F_ACC_RDONLY, H5P_DEFAULT);

	//打开数据集
	data_id = H5Dopen(file_id, datasetName, H5P_DEFAULT);

	//读取数据到data
	status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

	//把data写入到kernalWeight中
	for (int n = 0; n < outChannels; n++) {
		for (int c = 0; c < kernalSize.c; c++) {
			for (int h = 0; h < kernalSize.h; h++) {
				for (int w = 0; w < kernalSize.w; w++) {
					kernalWeight[n][c][h][w] = data[h * kernalSize.w * kernalSize.c * outChannels + w * kernalSize.c * outChannels + c * outChannels + n];
				}
			}
		}
	}

	//关闭数据文件和h5文件
	status = H5Dclose(data_id);
	status = H5Fclose(file_id);

	//释放内存
	delete[] data;
}



//读取二维数据：全连接层权值
void readWeight(const char* HDF5filename, const char* datasetName, nSize weightSize, float** weight) {

	float* data = new float[weightSize.h * weightSize.w];

	hid_t file_id;
	herr_t status;
	hid_t data_id;

	//打开h5文件
	file_id = H5Fopen(HDF5filename, H5F_ACC_RDONLY, H5P_DEFAULT);

	//打开数据集
	data_id = H5Dopen(file_id, datasetName, H5P_DEFAULT);

	//读取数据到data
	status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

	//把data写入到Weight中
	for (int h = 0; h < weightSize.h; h++) {
		for (int w = 0; w < weightSize.w; w++) {
			weight[h][w] = data[h * weightSize.w + w];
		}
	}

	//关闭数据文件和h5文件
	status = H5Dclose(data_id);
	status = H5Fclose(file_id);

	//释放内存
	delete[] data;
}