#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//device specific, TODO: make portable
#define MAX_THREAD_PER_BLOCK 1024

#define TRAIN_KERNEL_THREADS 1024


#include <memory.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define LENGTH_KERNEL	5

#define LENGTH_FEATURE0	32
//lf1 = 28
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
//lf2 = 14
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)
//lf3 = 10
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
//lf4 = 5
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)
//lf5 = 1
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define OUTPUT          10

#define ALPHA 0.5
#define PADDING 2

typedef unsigned char uint8;
typedef uint8 image[28][28];
//uint8 is unsigned char
//image is unsigned char[28][28]

typedef struct LeNet5
{
	double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];				//[1][6][5][5]
	double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];				//[6][16][5][5]
	double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];				//[16][120][5][5]
	double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];		//[120][10]

	double bias0_1[LAYER1];			//[6]
	double bias2_3[LAYER3];			//[16]
	double bias4_5[LAYER5];			//[120]
	double bias5_6[OUTPUT];			//[10]

}LeNet5;

typedef struct Feature
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];		//[1][32][32]
	double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];	//[6][28][28]
	double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];	//[6][14][14]
	double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];	//[16][10][10]
	double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];	//[16][5][5]
	double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];	//[120][1][1]
	double output[OUTPUT];										//[10]
}Feature;

//void TrainBatch(LeNet5* lenet, image* inputs, uint8* labels, int batchSize);
//void Train(LeNet5* lenet, image input, uint8 label);
//uint8 Predict(LeNet5* lenet, image input, uint8 count);
//void Initial(LeNet5* lenet);
void HANDLE_ERROR(cudaError_t cudaStatus);
void HANDLE_ERROR(cudaError_t cudaStatus) {
	if (cudaStatus != cudaError::cudaSuccess) {
		printf("Cuda Error : %s\n", cudaGetErrorString(cudaStatus));
	}
}



#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

//main function which applies convolution
//input and output are mapped according with stride=1 and  no padding
//e.g : 32x32 apply 5x5 kernel => 28x28 output
#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

/**/
#define cudaCONVOLUTE_VALID(tidx,input,output,weight)									\
{																						\
	int _conv_valid_tidx = GETLENGTH(output)*GETLENGTH(*(output));						\
	int iter_tidx = tidx,o0,o1;															\
	while(iter_tidx < _conv_valid_tidx){												\
		o0 = iter_tidx / GETLENGTH(*(output));											\
		o1 = iter_tidx % GETLENGTH(*(output));											\
		FOREACH(w0,GETLENGTH(weight))													\
			FOREACH(w1,GETLENGTH(*(weight))){											\
				(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];		\
			}																			\
		iter_tidx += TRAIN_KERNEL_THREADS;												\
	}																					\
}

#define cudaCONVOLUTION_FORWARD(tidx,input,output,weight,bias,action)										\
{																											\
	const int m1 = GETLENGTH(weight), m2 = GETLENGTH(*weight) , m3 = GETLENGTH(output[0]);					\
	const int m4 = GETLENGTH(*(output[0])), m5 = GETLENGTH(weight[0][0]), m6 = GETLENGTH(*(weight[0][0]));	\
	const int _conv_valid_tidx = m1 * m2 * m3 * m4 * m5 * m6;												\
	int iter_idx = tidx, x, y, o0, o1, w0, w1;																\
	while (iter_idx < _conv_valid_tidx) {																	\
		x = iter_idx / (m2 * m3 * m4 * m5 * m6);															\
		y = (iter_idx % (m2 * m3 * m4 * m5 * m6)) / (m3 * m4 * m5 * m6);									\
		o0 = (iter_idx % (m3 * m4 * m5 * m6)) / (m4 * m5 * m6);												\
		o1 = (iter_idx % (m4 * m5 * m6)) / (m5 * m6);														\
		w0 = (iter_idx % (m5 * m6)) / m6;																	\
		w0 = iter_idx % m6;																					\
		(output[y])[o0][o1] = atomicAdd_block(&((output[y])[o0][o1]),(double)((input[x])[o0 + w0][o1 + w1] * (weight[x][y])[w0][w1]));\
		iter_idx += TRAIN_KERNEL_THREADS;																	\
	}																										\
	__syncthreads();																						\
	int _conv_forward_tidx = GETLENGTH(output)*GETCOUNT(output[0]);											\
	int iter_tidx = tidx,j,i;																				\
	while(iter_tidx < _conv_forward_tidx){																	\
		j = iter_tidx % GETLENGTH(output);																	\
		i = iter_tidx/GETLENGTH(output);																	\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);								\
		iter_tidx += TRAIN_KERNEL_THREADS;																	\
	}																										\
}

/**/
//i0, i1 just iterate over feature map
#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

/*
for (int x = 0; x < GETLENGTH(weight); ++x)									
	for (int y = 0; y < GETLENGTH(*weight); ++y)							
		FOREACH(o0,GETLENGTH(output[y]))						
			FOREACH(o1,GETLENGTH(*(output[y])))					
				FOREACH(w0,GETLENGTH(weight[x][y]))					
					FOREACH(w1,GETLENGTH(*(weight[x][y])))
						//atomicAdd_block((output[y])[o0][o1],(input[x])[o0 + w0][o1 + w1] * (weight[x][y])[w0][w1]);
						(output[y])[o0][o1] += (input[x])[o0 + w0][o1 + w1] * (weight[x][y])[w0][w1]; 
*/
/*
const int m1 = GETLENGTH(weight), m2 = GETLENGTH(*weight) , m3 = GETLENGTH(output[0]);
const int m4 = GETLENGTH(*(output[0])), m5 = GETLENGTH(weight[0][0]) , m6 = GETLENGTH(*(weight[0][0]));
const int _conv_valid_tidx = m1*m2*m3*m4*m5*m6;
int iter_idx = tidx,x,y,o0,o1,w0,w1;
while(iter_idx < _conv_valid_tidx){
	x = iter_idx / (m2*m3*m4*m5*m6);
	y = (iter_idx % (m2*m3*m4*m5*m6)) / (m3*m4*m5*m6);
	o0 = (iter_idx % (m3*m4*m5*m6)) / (m4*m5*m6);
	o1 = (iter_idx % (m4*m5*m6)) / (m5*m6);
	w0 = (iter_idx % (m5*m6)) / m6;
	w0 = iter_idx % m6;
	atomicAdd_block((output[y])[o0][o1],(input[x])[o0 + w0][o1 + w1] * (weight[x][y])[w0][w1]);
	iter_idx += TRAIN_KERNEL_THREADS;
}
*/

//fl1 : 32x32 (use 6 convolution)-> 28x28x6 : weight[1][6][5][5]
//features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, relu
#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			CONVOLUTE_VALID(input[x], output[y], weight[x][y]);					\
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

//len0 = len1 = 2
#define cudaSUBSAMP_MAX_FORWARD(tidx,input,output)												\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	const int m1 =  GETLENGTH(output), m2 = GETLENGTH(*(output)) , m3 = GETLENGTH(**(output));	\
	const int _sub_samp_tidx = m1*m2*m3;														\
	int iter_tidx = tidx,i,o0,o1;																\
	while(iter_tidx < _sub_samp_tidx)															\
	{																							\
		i = iter_tidx / (m2*m3);																\
		o0 = (iter_tidx%(m2*m3)) / m3;															\
		o1 = iter_tidx%m3;																		\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
		iter_tidx += TRAIN_KERNEL_THREADS;														\
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}
////[120][1][1], [10][10],      [120][10],        [10]
//(ff.layer5, ff.output, lenet->weight5_6, lenet->bias5_6, relu);
#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
}

#define cudaDOT_PRODUCT_FORWARD(tidx,input,output,weight,bias,action)				\
{																					\
	const int _dot_prod_tidx = GETLENGTH(*weight);									\
	int iter_tidx = tidx , y;														\
	while(iter_tidx <  _dot_prod_tidx){												\
		y = iter_tidx;																\
		for (int x = 0; x < GETLENGTH(weight); ++x)									\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];			\
		iter_tidx += TRAIN_KERNEL_THREADS;											\
	}																				\
	__syncthreads();																\
	if(tidx < GETLENGTH(bias))														\
		((double *)output)[tidx] = action(((double *)output)[tidx] + bias[tidx]);	\
}

/*
#define cudaDOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																				\
																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}
*/

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}

__device__ double relu(double x)
{
	return x * (x > 0);
}

__device__ double relugrad(double y)
{
	return y > 0;
}

__device__ void forward(LeNet5* lenet, Feature* features, double(*action)(double))
{
	//1x32x32 (use 6 convolution of 5x5)-> 6x28x28
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

__device__ void backward(LeNet5* lenet, LeNet5* deltas, Feature* errors, Feature* features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

//image* input -> [28][28] convert to features->input[0][32][32]
//with padding = 2
__device__ void load_input(Feature* features, image input)
{
	const long sz = 784; //28x28
	double mean = 0, std = 0;

	FOREACH(j, 28)
		FOREACH(k, 28)
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean * mean);
	//for j : [0 , 28)
	//for k : [0 , 28)
	
	FOREACH(j, 28)
		FOREACH(k, 28)
	{
		//z-score normalization
		features->input[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}

__device__ void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

__device__ void load_target(Feature* features, Feature* errors, int label)
{
	double* output = (double*)features->output;
	double* error = (double*)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

__device__ uint8 get_result(Feature* features, uint8 count)
{
	double* output = (double*)features->output;
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

__device__ double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		//srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)5 << i;
	lvalue |= (unsigned long long)5 >> -i;
	return *(double*)&lvalue - 3;
}


__global__ void train_kernel(LeNet5* lenet, image* inputs, uint8* labels, int startidx, int batchSize, double* buffer);
__global__ void update_weights_kernel(LeNet5* lenet, double* buffer, int k);


//<<<batchSize,1>>> TODO: <<<batchSize, MAX_THREAD_PER_BLOCK>>>
__global__ void train_kernel(LeNet5* lenet, image* inputs, uint8* labels, int startidx, int batchSize, double* buffer) {

	int i = blockIdx.x;//[0,299]
	int tidx = threadIdx.x;
	//printf("train_kernel blockidx: %d\n", i);

	__shared__ Feature* features;
	__shared__ Feature* errors;
	__shared__ LeNet5* deltas;

	if (tidx == 0) {
		features = (Feature*)malloc(sizeof(Feature));
		errors = (Feature*)malloc(sizeof(Feature));
		deltas = (LeNet5*)malloc(sizeof(LeNet5));
		if (features == nullptr || errors == nullptr || deltas == nullptr) {
			printf("ERROR!train_kernel: bidx: %d , tidx: %d heap memory cant be allocated\n");
		}
	}
	__syncthreads(); //wait for malloc

	////LOAD INPUT - 1024 threads
	if (tidx == 0)
		load_input(features, inputs[startidx + i]); //784 operations : reduc sum slows down, dont use that
	__syncthreads();

	///FORWARD: forward(lenet, features, relu);
	cudaCONVOLUTION_FORWARD(tidx, features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, relu);
	__syncthreads();
	cudaSUBSAMP_MAX_FORWARD(tidx, features->layer1, features->layer2);
	__syncthreads();
	cudaCONVOLUTION_FORWARD(tidx, features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, relu);
	__syncthreads();
	cudaSUBSAMP_MAX_FORWARD(tidx, features->layer3, features->layer4);
	__syncthreads();
	cudaCONVOLUTION_FORWARD(tidx, features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, relu);
	__syncthreads();
	cudaDOT_PRODUCT_FORWARD(tidx, features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, relu);
	__syncthreads();

	///LOAD TARGET - 100 operations dont parallalize
	if (tidx == 0)
		load_target(features, errors, labels[startidx+i]);
	__syncthreads();

	///BACKWARD: backward(lenet, deltas, errors, features, relugrad);
	if (tidx == 0)
	{
		//DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, relugrad);
		//CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, relugrad);
		//SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
		//CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, relugrad);
		//SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
		//CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, relugrad);
	}
	__syncthreads();
	
	{
		//every block needs to do this
		//buffer[] is shared among blocks
		int iter_tidx = tidx;
		while (iter_tidx < GETCOUNT(LeNet5)) {
			//buffer[iter_tidx] += ((double*)deltas)[iter_tidx]; //atomic add
			buffer[iter_tidx] = atomicAdd( buffer + iter_tidx , ((double*)deltas)[iter_tidx]); //atomic add
			iter_tidx += TRAIN_KERNEL_THREADS;
		}
	}

	__syncthreads();
	if (tidx == 0) {
		free(features);
		free(errors);
		free(deltas);
	}
}


//inputs = &train_data[i] , labels = &train_labels[i], batchsize = 300
void TrainBatch(LeNet5* lenet, image* inputs, uint8* labels, int startidx, int batchSize)
{
	//buffer on device
	double* buffer;
	HANDLE_ERROR(cudaMalloc((void**)&buffer, sizeof(LeNet5)));
	HANDLE_ERROR(cudaMemset(buffer, 0, sizeof(LeNet5)));
	
	int i = 0;
	
	//launch kernel here with blocks = batch_size
	train_kernel <<<batchSize, TRAIN_KERNEL_THREADS >>> (lenet, inputs, labels, startidx, batchSize, buffer);

	HANDLE_ERROR(cudaGetLastError());
	//HANDLE_ERROR(cudaDeviceSynchronize());
	//launch seprate kernel for this operation
	//lenet is also shared among blocks
	//this needs to be filled parallely

	double k = ALPHA / batchSize;
	//51*1024 = 52224 > (GETCOUNT(LeNet5) = 51902)
	update_weights_kernel <<<51,1024>>> (lenet, buffer, k);
	
	//HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaFree(buffer));
}

__global__ void update_weights_kernel(LeNet5* lenet, double* buffer, int k) {
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < GETCOUNT(LeNet5))
		((double*)lenet)[i] += k * buffer[i];
}

__device__ uint8 Predict(LeNet5* lenet, image input, uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	return get_result(&features, count);
}

__global__ void Initial(LeNet5* lenet)
{
	//if (blockIdx.x == 0 && threadIdx.x == 0) {
		//can be done parllely, since lenet must be in device memory
		for (double* pos = (double*)lenet->weight0_1; pos < (double*)lenet->bias0_1; *pos++ = f64rand());
		for (double* pos = (double*)lenet->weight0_1; pos < (double*)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
		for (double* pos = (double*)lenet->weight2_3; pos < (double*)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
		for (double* pos = (double*)lenet->weight4_5; pos < (double*)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
		for (double* pos = (double*)lenet->weight5_6; pos < (double*)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
		for (int* pos = (int*)lenet->bias0_1; pos < (int*)(lenet + 1); *pos++ = 0);
	//}
}
