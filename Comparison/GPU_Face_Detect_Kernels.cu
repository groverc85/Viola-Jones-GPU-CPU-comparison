#include "GPU_Face_Detect.cuh"
#include "cuda.h"
#include "lock.h"

#define CONSTANT_MEM_SIZE 32
__constant__ GPUHaarStageClassifier stageClassifiers[CONSTANT_MEM_SIZE];

texture<int, 2, cudaReadModeElementType> sumImageRef;
texture<float, 2, cudaReadModeElementType> sqSumImageRef;

void allocateGPUCascade(GPUHaarCascade &h_gpuCascade, GPUHaarCascade &dev_gpuCascade)
{
	// 变量拷贝
	dev_gpuCascade.flags = h_gpuCascade.flags;
	dev_gpuCascade.numOfStages = h_gpuCascade.numOfStages;
	dev_gpuCascade.orig_window_size = h_gpuCascade.orig_window_size;
	dev_gpuCascade.real_window_size = h_gpuCascade.real_window_size;
	dev_gpuCascade.img_window_size = h_gpuCascade.img_window_size;
	dev_gpuCascade.scale = h_gpuCascade.scale;
	dev_gpuCascade.totalNumOfClassifiers = h_gpuCascade.totalNumOfClassifiers;

	// 为设备classifiers分配空间，并从host中拷贝classifiers
	size_t GPU_Classifier_Size = h_gpuCascade.totalNumOfClassifiers * sizeof(GPUHaarClassifier);
	HANDLE_ERROR( cudaMalloc( (void**)&dev_gpuCascade.haar_classifiers, GPU_Classifier_Size ) );
	HANDLE_ERROR( cudaMemcpy(dev_gpuCascade.haar_classifiers, h_gpuCascade.haar_classifiers, GPU_Classifier_Size, cudaMemcpyHostToDevice));

	HANDLE_ERROR( cudaMalloc( (void**)&dev_gpuCascade.scaled_haar_classifiers, GPU_Classifier_Size ) );
	HANDLE_ERROR( cudaMemcpy(dev_gpuCascade.scaled_haar_classifiers, h_gpuCascade.scaled_haar_classifiers, GPU_Classifier_Size, cudaMemcpyHostToDevice));

	if(h_gpuCascade.numOfStages > CONSTANT_MEM_SIZE)
	{
		printf("ERROR: Number of stages is larger than the max size of constant memory alloted");
		system("pause");
		return;
	}

	HANDLE_ERROR( cudaMemcpyToSymbol( stageClassifiers, h_gpuCascade.haar_stage_classifiers, sizeof(GPUHaarStageClassifier) * h_gpuCascade.numOfStages ) );
}

void allocateIntegralImagesGPU(CvMat * sumImage, CvMat *sqSumImage, cudaArray *dev_sumArray, cudaArray * dev_sqSumArray)
{
	//==================================================================
	//分配texture memory用于计算sum integal image
	//==================================================================

	// 创建纹理的channel描述（1 channel 32 bits, signed int类型)
	cudaChannelFormatDesc sum_channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned );

	// 为sum image texture分配设备内存
	HANDLE_ERROR( cudaMallocArray(&dev_sumArray, &sum_channelDesc, sumImage->width, sumImage->height));

	// 从OpenCv拷贝图像数据到设备内存中
	HANDLE_ERROR(cudaMemcpy2DToArray(dev_sumArray, 0, 0, sumImage->data.i, sumImage->step, sumImage->width * sizeof(int), sumImage->height, cudaMemcpyHostToDevice));
	
	// 设置CUDA纹理引用的参数
	sumImageRef.addressMode[0] = cudaAddressModeWrap;
	sumImageRef.addressMode[1] = cudaAddressModeWrap;
	sumImageRef.filterMode = cudaFilterModePoint; //cudaFilterModeLinear
	sumImageRef.normalized = false;

	// 绑定纹理引用到分配的设备内存上
	HANDLE_ERROR( cudaBindTextureToArray(sumImageRef, dev_sumArray, sum_channelDesc));

	//==================================================================
	//分配texture memory用于计算squared sum integal image
	//==================================================================

	// 创建纹理的channel描述（1 channel 64 bits, signed int类型)
	cudaChannelFormatDesc sqSum_channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	
	// 为sum image texture分配设备内存
	HANDLE_ERROR( cudaMallocArray(&dev_sqSumArray, &sqSum_channelDesc, sqSumImage->width, sqSumImage->height));

	// 从OpenCv拷贝图像数据到设备内存中
	HANDLE_ERROR(cudaMemcpy2DToArray(dev_sqSumArray, 0, 0, sqSumImage->data.fl, sqSumImage->step, sqSumImage->width * sizeof(int), sqSumImage->height, cudaMemcpyHostToDevice));
	
	// 设置CUDA纹理引用的参数
	sqSumImageRef.addressMode[0] = cudaAddressModeWrap;
	sqSumImageRef.addressMode[1] = cudaAddressModeWrap;
	sqSumImageRef.filterMode = cudaFilterModeLinear;
	sqSumImageRef.normalized = false;

	// 绑定纹理引用到分配的设备内存上
	HANDLE_ERROR( cudaBindTextureToArray(sqSumImageRef, dev_sqSumArray, sqSum_channelDesc));
}

void releaseTextures()
{
	cudaUnbindTexture(sumImageRef);
	cudaUnbindTexture(sqSumImageRef);
}

__device__ float calculateMean(GPURect rect)
{
	int A = tex2D(sumImageRef, rect.x, rect.y);
	int B = tex2D(sumImageRef, rect.x + rect.width, rect.y);
	int C = tex2D(sumImageRef, rect.x + rect.width, rect.y + rect.height);
	int D = tex2D(sumImageRef, rect.x, rect.y + rect.height);

	return (float)(A - B + C - D);
}

__device__ float calculateSum(GPURect rect, int win_start_x, int win_start_y)
{
	float tx = win_start_x + rect.x;
	float ty = win_start_y + rect.y;

	int A = tex2D(sumImageRef, tx, ty);
	int B = tex2D(sumImageRef, tx + rect.width, ty);
	int C = tex2D(sumImageRef, tx + rect.width, ty + rect.height);
	int D = tex2D(sumImageRef, tx, ty + rect.height);

	return (float)(A - B + C - D);
}

__device__ int getOffset(int x, int y)
{
	return x + y * blockDim.x * gridDim.x;
}

__device__ float runHaarFeature(GPUHaarClassifier classifier, GPURect detectionWindow, float variance_norm_factor, float weightScale)
{
	double t = classifier.threshold * variance_norm_factor;

	double sum = calculateSum(classifier.haar_feature.rect0.r, detectionWindow.x, detectionWindow.y) * classifier.haar_feature.rect0.weight * weightScale;
	sum += calculateSum(classifier.haar_feature.rect1.r, detectionWindow.x, detectionWindow.y) * classifier.haar_feature.rect1.weight * weightScale;

	// 若存在第三个矩形
	if(classifier.haar_feature.rect2.weight)
		sum += calculateSum(classifier.haar_feature.rect2.r, detectionWindow.x, detectionWindow.y) * classifier.haar_feature.rect2.weight * weightScale;
            
	if(sum >= t)
		return classifier.alpha1;
	else
		return classifier.alpha0;
}

__device__ float calculateVariance(GPURect detectionWindow)
{
	float inv_window_area = 1.0f / ((float)detectionWindow.width * detectionWindow.height);
	float weightScale = inv_window_area;

	// HaarCascade file requires normalization of features
	float mean = calculateMean(detectionWindow) * inv_window_area;
	
	float variance_norm_factor = tex2D(sqSumImageRef, detectionWindow.x, detectionWindow.y) - 
		tex2D(sqSumImageRef, detectionWindow.x + detectionWindow.width, detectionWindow.y) - 
		tex2D(sqSumImageRef, detectionWindow.x, detectionWindow.y + detectionWindow.height) + 
		tex2D(sqSumImageRef, detectionWindow.x + detectionWindow.width, detectionWindow.y + detectionWindow.height);
		
	variance_norm_factor = variance_norm_factor * inv_window_area - mean * mean;
	//variance_norm_factor = sqrt(variance_norm_factor * inv_window_area - mean * mean);

	if(variance_norm_factor >= 0.0f)
		variance_norm_factor = sqrt(variance_norm_factor);
	else
	{		
		variance_norm_factor = 1.0f;
	}

	return variance_norm_factor;
}

//==================================================================
// 版本4：该部分通过将所有的检测窗口分配给block内的threads来达到使GPU保持忙碌的目的。每个阶段
// 之后检测窗口的数目都会减少，剩下的检测窗口是上一个阶段判断为可能是脸部区域的窗口。这一步一直
// 是脸部区域，或者持续到所有的剩余窗口都全部都已经被排除
//==================================================================

__global__ void haarDetection_v4(GPUHaarCascade haarCascade, GPURect * detectedFaces, Lock * locks)
{
	// Load Detection Windows into shared memory
	__shared__ GPURect detectionWindows[THREADS_PER_BLOCK_V4];

	int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;

	detectionWindows[threadIndex].x = threadIdx.x + blockIdx.x * blockDim.x;
	detectionWindows[threadIndex].y = threadIdx.y + blockIdx.y * blockDim.y;
	detectionWindows[threadIndex].width = haarCascade.real_window_size.width;
	detectionWindows[threadIndex].height = haarCascade.real_window_size.height;

	__shared__ int numOfWindowsLeft; 
	numOfWindowsLeft = THREADS_PER_BLOCK_V4;

	__syncthreads();

	for(int i = 0; i < haarCascade.numOfStages; i++)
	{
		if(numOfWindowsLeft == 0)
			break;

		// passedWindows用于存储通过该阶段的所有窗口
		__shared__ GPURect passedWindows[THREADS_PER_BLOCK_V4];
		passedWindows[threadIndex].width = 0.0f;

		// numOfWindowsPassed用于寄出已经通过该stage的窗口数目
		__shared__ int numOfWindowsPassed;
		numOfWindowsPassed = 0;

		int windowCounter = 0;

		__syncthreads();

		while(windowCounter < numOfWindowsLeft)
		{
			// 每个块内有256个线程。若有8个检测窗口，则每个检测窗口32个线程。
			int currentWindow_Index = threadIndex / WARP_SIZE;
			GPURect currentDetectionWindow = detectionWindows[windowCounter + currentWindow_Index];

			__shared__ float stage_sum[THREADS_PER_BLOCK_V4];

			float variance_norm_factor = calculateVariance(currentDetectionWindow);
			float inv_window_area = 1.0f / ((float)currentDetectionWindow.width * currentDetectionWindow.height);

			int featureIdx = threadIndex % WARP_SIZE;
			float featureSum = 0.0f;

			while(featureIdx < stageClassifiers[i].numofClassifiers)
			{
				int index = featureIdx + stageClassifiers[i].classifierOffset;
				GPUHaarClassifier classifier = haarCascade.scaled_haar_classifiers[index];
				
				featureSum += runHaarFeature(classifier,  currentDetectionWindow, variance_norm_factor, inv_window_area);

				featureIdx += WARP_SIZE;
			}

			stage_sum[threadIndex] = featureSum;
			__syncthreads();

			// 计算每个检测窗口的haar features的综合，并与阈值进行比较判断是否符合条件。
			int j = WARP_SIZE/2;
			while(j != 0)
			{
				if(threadIndex % WARP_SIZE < j)
					stage_sum[threadIndex] += stage_sum[threadIndex + j];

				__syncthreads();
				j /= 2;
			}

			// 每个warp的第一个线程检测该warp下的检测窗口是否通过
			if(threadIndex % WARP_SIZE == 0)
			{
				// 若通过，则保存该检测窗口
				if(stage_sum[threadIndex] > stageClassifiers[i].threshold && currentDetectionWindow.width != 0)
				{
					locks[blockIdx.x + blockIdx.y * gridDim.x].lock();
					numOfWindowsPassed += 1;
					passedWindows[numOfWindowsPassed - 1] = currentDetectionWindow;
					locks[blockIdx.x + blockIdx.y * gridDim.x].unlock();
				}
			}

			windowCounter += DETECTION_WINDOW_STRIDE_V4;

			__syncthreads();
		}

		// 将通过信息拷贝到detectionWindows queue中
		detectionWindows[threadIndex] = passedWindows[threadIndex];

		if(threadIndex == 0)
		{
			numOfWindowsLeft = numOfWindowsPassed;
		}

		__syncthreads();
	}

	// 拷贝detectionWindow结果到output中
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = getOffset(x,y);

	detectedFaces[offset] = detectionWindows[threadIndex];
}

//===============================================================================
// 版本3：Viola & Jones原始实现，即如果一个Classifier fail，抛弃整个cascade（）
//===============================================================================

__global__ void haarDetection_v3(GPUHaarCascade haarCascade, GPURect * detectedFaces)
{
	// for start thread + blocksize offset to next thread < w * h
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for(; index < haarCascade.img_detection_size.width * haarCascade.img_detection_size.height; index += blockDim.x * gridDim.x)
	{
		struct GPURect detectionWindow;
		detectionWindow.x = (int)(index % haarCascade.img_detection_size.width);
		detectionWindow.y = (int)(index / haarCascade.img_detection_size.width);
		detectionWindow.width = haarCascade.real_window_size.width;
		detectionWindow.height = haarCascade.real_window_size.height;

		float variance_norm_factor = calculateVariance(detectionWindow);
		float inv_window_area = 1.0f / (detectionWindow.width * detectionWindow.height);
		float weightScale = inv_window_area;

		bool faceDetected = true;

		for(int i = 0; i < haarCascade.numOfStages; i++)
		{
			float stage_sum = 0.0;
			for(int j = 0; j < stageClassifiers[i].numofClassifiers; j++)
			{
				int index = j + stageClassifiers[i].classifierOffset;
				GPUHaarClassifier classifier = haarCascade.scaled_haar_classifiers[index];
				stage_sum += runHaarFeature(classifier,  detectionWindow, variance_norm_factor, inv_window_area);
			}

			// 若Classifier没有通过，就抛弃整个cascade
			if( stage_sum < stageClassifiers[i].threshold)
			{
				faceDetected = false;
				break;
			}
		}

		// 传递检测窗口的输出到global memory中，使得host可以获得该部分数据
		if(faceDetected)
			detectedFaces[getOffset(detectionWindow.x, detectionWindow.y)] = detectionWindow;
	}
}

//===============================================================================
// 版本2：使得每个检测窗口都强行计算所有的Haar特征。使用shared memory来load每个阶段的classifiers
//===============================================================================

__global__ void haarDetection_v2(GPUHaarCascade haarCascade, GPURect * detectedFaces)
{
	#define THREADS_PER_BLOCK 256

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = getOffset(x,y);

	// 若当前像素越界，直接返回
	GPURect detectionWindow;
	detectionWindow.x = x;
	detectionWindow.y = y;
	detectionWindow.width = haarCascade.real_window_size.width;
	detectionWindow.height = haarCascade.real_window_size.height;

	float variance_norm_factor = calculateVariance(detectionWindow);
	float inv_window_area = 1.0f / ((float)detectionWindow.width * detectionWindow.height);
	float weightScale = inv_window_area;

	bool faceDetected = true;

	for(int i = 0; i < haarCascade.numOfStages; i++)
	{
		// 将该阶段所有的classifiers拷贝到shared memory中来加速
		__shared__ GPUHaarClassifier sharedClassifiers[THREADS_PER_BLOCK];

		int threadIndex =  threadIdx.x + threadIdx.y * blockDim.x;
		int stageOffset = stageClassifiers[i].classifierOffset;

		if(threadIndex + stageOffset < haarCascade.totalNumOfClassifiers)
			sharedClassifiers[threadIndex] = haarCascade.scaled_haar_classifiers[threadIndex + stageOffset];

		__syncthreads();

		float stage_sum = 0.0;
		for(int j = 0; j < stageClassifiers[i].numofClassifiers; j++)
		{
			GPUHaarClassifier classifier = sharedClassifiers[j];

			stage_sum += runHaarFeature(classifier,  detectionWindow, variance_norm_factor, inv_window_area);
		}

		bool stagePassed = stage_sum > stageClassifiers[i].threshold;
		faceDetected = stagePassed && faceDetected;
	}
	
	// 传递检测窗口的输出到global memory中，使得host可以获得该部分数据
	if(faceDetected)
		detectedFaces[offset] = detectionWindow;
	
}

//===============================================================================
// 版本1：Viola & Jones原始实现，即如果一个Classifier fail，抛弃整个cascade
//===============================================================================

__global__ void haarDetection_v1(GPUHaarCascade haarCascade, GPURect * detectedFaces)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = getOffset(x,y);

	// 若当前像素越界，直接返回	
	if(x < haarCascade.img_detection_size.width || y < haarCascade.img_detection_size.height)
	{
		GPURect detectionWindow;
		detectionWindow.x = x;
		detectionWindow.y = y;
		detectionWindow.width = haarCascade.real_window_size.width;
		detectionWindow.height = haarCascade.real_window_size.height;

		float variance_norm_factor = calculateVariance(detectionWindow);
		float inv_window_area = 1.0f / ((float)detectionWindow.width * detectionWindow.height);
		float weightScale = inv_window_area;

		// 假设脸部被检测到
		detectedFaces[offset] = detectionWindow;

		// cascade的每个阶段
		for(int i = 0; i < haarCascade.numOfStages; i++)
		{
			float stage_sum = 0.0;
			for(int j = 0; j < stageClassifiers[i].numofClassifiers; j++)
			{
				int index = j + stageClassifiers[i].classifierOffset;
				GPUHaarClassifier classifier = haarCascade.scaled_haar_classifiers[index];

				stage_sum += runHaarFeature(classifier, detectionWindow, variance_norm_factor, inv_window_area);
			}

			// 若Classifier fail，抛弃整个cascade
			if( stage_sum < stageClassifiers[i].threshold)
			{
				// Set width to zero to indicate on CPU side that this is not a face
				detectedFaces[offset].width = 0;
				break;
			}
		}
	}
}

