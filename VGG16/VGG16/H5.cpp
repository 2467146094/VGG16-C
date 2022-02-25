#include"H5.h"
#include"hdf5.h"
#include"cnn.h"

//��ȡһά���ݣ�ƫ��
void readWeight(const char* HDF5filename, const char* datasetName, float* bias) {

	hid_t file_id;
	herr_t status;
	hid_t data_id;

	//��h5�ļ�
	file_id = H5Fopen(HDF5filename, H5F_ACC_RDONLY, H5P_DEFAULT);

	//�����ݼ�
	data_id = H5Dopen(file_id, datasetName, H5P_DEFAULT);

	//��ȡ���ݵ�data
	status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, bias);

	//�ر������ļ���h5�ļ�
	status = H5Dclose(data_id);
	status = H5Fclose(file_id);
}



//��ȡ��ά���ݣ������Ȩֵ
void readWeight(const char* HDF5filename, const char* datasetName, nSize kernalSize, int outChannels, float**** kernalWeight) {

	float* data = new float[kernalSize.h * kernalSize.w * kernalSize.c * outChannels];

	hid_t file_id;
	herr_t status;
	hid_t data_id;

	//��h5�ļ�
	file_id = H5Fopen(HDF5filename, H5F_ACC_RDONLY, H5P_DEFAULT);

	//�����ݼ�
	data_id = H5Dopen(file_id, datasetName, H5P_DEFAULT);

	//��ȡ���ݵ�data
	status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

	//��dataд�뵽kernalWeight��
	for (int n = 0; n < outChannels; n++) {
		for (int c = 0; c < kernalSize.c; c++) {
			for (int h = 0; h < kernalSize.h; h++) {
				for (int w = 0; w < kernalSize.w; w++) {
					kernalWeight[n][c][h][w] = data[h * kernalSize.w * kernalSize.c * outChannels + w * kernalSize.c * outChannels + c * outChannels + n];
				}
			}
		}
	}

	//�ر������ļ���h5�ļ�
	status = H5Dclose(data_id);
	status = H5Fclose(file_id);

	//�ͷ��ڴ�
	delete[] data;
}



//��ȡ��ά���ݣ�ȫ���Ӳ�Ȩֵ
void readWeight(const char* HDF5filename, const char* datasetName, nSize weightSize, float** weight) {

	float* data = new float[weightSize.h * weightSize.w];

	hid_t file_id;
	herr_t status;
	hid_t data_id;

	//��h5�ļ�
	file_id = H5Fopen(HDF5filename, H5F_ACC_RDONLY, H5P_DEFAULT);

	//�����ݼ�
	data_id = H5Dopen(file_id, datasetName, H5P_DEFAULT);

	//��ȡ���ݵ�data
	status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

	//��dataд�뵽Weight��
	for (int h = 0; h < weightSize.h; h++) {
		for (int w = 0; w < weightSize.w; w++) {
			weight[h][w] = data[h * weightSize.w + w];
		}
	}

	//�ر������ļ���h5�ļ�
	status = H5Dclose(data_id);
	status = H5Fclose(file_id);

	//�ͷ��ڴ�
	delete[] data;
}