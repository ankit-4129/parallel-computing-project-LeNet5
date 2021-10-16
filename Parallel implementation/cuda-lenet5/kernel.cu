#include "lenet.h"


#define FILE_TRAIN_IMAGE		"./mnist_dataset/train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"./mnist_dataset/train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"./mnist_dataset/t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"./mnist_dataset/t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000



int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
	FILE* fp_image = fopen(data_file, "rb");
	FILE* fp_label = fopen(label_file, "rb");
	if (!fp_image || !fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data) * count, 1, fp_image);
	fread(label, count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

int save(LeNet5* lenet, char filename[])
{
	FILE* fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int load(LeNet5* lenet, char filename[])
{
	FILE* fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

void load_dataset(image* train_data, uint8* train_label, image* test_data, uint8* test_label) {

	if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the exe\n");
		free(train_data);
		free(train_label);
		system("pause");
	}
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the exe\n");
		free(test_data);
		free(test_label);
		system("pause");
	}
}

void display_img(image img, int binary) {
	char binch[] = { ' ','0' };
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (binary)
				printf("%c ", binch[(img[i][j] > 0)]);
			else
				printf("%3d ", (int)img[i][j]);
		}
		printf("\n");
		if (!binary)
			printf("\n");
	}
	printf("\n");
}

//batchsize = 300, total_size = 60000
void training(LeNet5* lenet, image* train_data, uint8* train_label, int batch_size, int total_size)
{
	//must be sequential
	//int i = batch_size*2, percent = 0;

	//device set limit should be called once
	HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024));
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data, train_label, i, batch_size);
		if (i * 100 / total_size > percent)
			printf("batchsize: %d\ttrain: %2d%%\n", batch_size, percent = i * 100 / total_size);
	}
}

__device__ int testing(LeNet5* lenet, image* test_data, uint8* test_label, int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict(lenet, test_data[i], 10);
		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}


void train_test(image* train_data, uint8* train_label, image* test_data, uint8* test_label)
{	
	//make lenet on device and share among all blocks
	//i.e: only one instance of lenet will be on device
	LeNet5* dev_lenet;
	HANDLE_ERROR(cudaMalloc((void**)&dev_lenet, sizeof(LeNet5)));
	//Initial<<<1,1>>>(lenet);

	
	int batches = 300;
	training(dev_lenet, train_data, train_label, batches, COUNT_TRAIN);
	
	//TODO: testing
	/*
	int right = testing(lenet, test_data, test_label, COUNT_TEST);

	printf("%d/%d\n", right, COUNT_TEST);
	printf("Accuracy = %lf\n", right / (double)COUNT_TEST);
	//printf("Time: %lf seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC);
	/**/
	HANDLE_ERROR(cudaFree(dev_lenet));
}



int main()
{
	image* train_data = (image*)calloc(COUNT_TRAIN, sizeof(image));
	uint8* train_label = (uint8*)calloc(COUNT_TRAIN, sizeof(uint8));
	image* test_data = (image*)calloc(COUNT_TEST, sizeof(image));
	uint8* test_label = (uint8*)calloc(COUNT_TEST, sizeof(uint8));

	load_dataset(train_data, train_label, test_data, test_label);

	//load data from host to device
	image* dev_train_data;
	uint8* dev_train_label;
	image* dev_test_data;
	uint8* dev_test_label;
	//allocate on device
	HANDLE_ERROR(cudaMalloc((void**)&dev_train_data, COUNT_TRAIN * sizeof(image)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_train_label, COUNT_TRAIN * sizeof(uint8)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_test_data, COUNT_TEST * sizeof(image)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_test_label, COUNT_TEST * sizeof(uint8)));

	//copy to device
	HANDLE_ERROR(cudaMemcpy(dev_train_data, train_data, COUNT_TRAIN * sizeof(image), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_train_label, train_label, COUNT_TRAIN * sizeof(uint8), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_test_data, test_data, COUNT_TEST * sizeof(image), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_test_label, test_label, COUNT_TEST * sizeof(uint8), cudaMemcpyHostToDevice));

	//train_test(train_data, train_label, test_data, test_label);
	train_test(dev_train_data, dev_train_label, dev_test_data, dev_test_label);
	
	//TODO: free host memory after copying to device ASAP
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);
	
	HANDLE_ERROR(cudaFree(dev_train_data));
	HANDLE_ERROR(cudaFree(dev_train_label));
	HANDLE_ERROR(cudaFree(dev_test_data));
	HANDLE_ERROR(cudaFree(dev_test_label));
	/**/
	return 0;
}

