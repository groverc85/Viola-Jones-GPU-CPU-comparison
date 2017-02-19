
#include "GPU_Face_Detect.cuh"

#include "cuda.h"
#include "lock.h"

//===========================================================================
// 声明GPU脸部检测的变量
//===========================================================================

cudaEvent_t start, stop;

// 设备内存haar cascade
GPUHaarCascade h_gpuHaarCascade;
GPUHaarCascade dev_gpuHaarCascade;

// 声明GPU纹理内存的指针
cudaArray * dev_sumArray = NULL;
cudaArray * dev_sqSumArray = NULL;

// 数组用于拷贝GPU检测的脸部结果，用于CPU的后处理（显示等）
GPURect *detectedFaces, *dev_detectedFaces;
size_t detectedFacesSize;

// 设备内存初始化
void initGPU(GPUHaarCascade &gpuCascade, IplImage * image, CvMat *sumImg, CvMat *sqSumImg)
{
	int width = image->width;
	int height = image->height;

	//=========================================================================
	// 定义并初始化CUDA even timing来衡量表现
	//=========================================================================

	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );

	//=========================================================================
	// 定义GPU Haar Cascade structures并转换CvHaarCascade到GPU Haar Cascade structures
	//=========================================================================
	
	// load GPU haar cascade到host中
	h_gpuHaarCascade.load(&gpuCascade);

	// 分配设备内存
	allocateGPUCascade( h_gpuHaarCascade, dev_gpuHaarCascade);

	//==================================================================
	// 创建积分图像并拷贝到设备纹理内存中
	//==================================================================

	// 转换双精度square sum image到float京都
	cv::Mat sqSumMat(sqSumImg->rows, sqSumImg->cols, CV_64FC1, sqSumImg->data.fl);
	sqSumMat.convertTo(sqSumMat, CV_32FC1);
	CvMat float_sqsm = sqSumMat;
	
	// 为积分图像分配纹理内存并拷贝OpenCV(cvIntegral())结果到设备中
	allocateIntegralImagesGPU(sumImg, &float_sqsm, dev_sumArray, dev_sqSumArray);

	//===============================================================================
	// 分配并拷贝face array到设备中用于存储检测结果
	//==============================================================================

	// CPU内存分配
	detectedFacesSize = width * height * sizeof(GPURect);
	detectedFaces = (GPURect *)malloc(detectedFacesSize);
	memset(detectedFaces, 0, detectedFacesSize);

	// GPU内存分配以及host数据拷贝
	HANDLE_ERROR( cudaMalloc( (void**)&dev_detectedFaces, detectedFacesSize ) );
	HANDLE_ERROR( cudaMemcpy(dev_detectedFaces, detectedFaces, detectedFacesSize, cudaMemcpyHostToDevice));
}

// 根据gpuFaces数组，检查CvRect.width来判断GPU是否判定该窗口为脸部区域
int selectFaces(std::vector<CvRect> &faces, GPURect *gpuFaces, int pixels)
{
	int faces_detected = 0;
	for( int i = 0; i < pixels; i++ )
	{
		// 只提取检测到的矩阵
		GPURect face_rect = gpuFaces[i];

		if(face_rect.width != 0)
		{
			CvRect convertRect(face_rect.x, face_rect.y, face_rect.width, face_rect.height);
			faces.push_back(convertRect);
			faces_detected++;
		}
	}

	return faces_detected;
}

void startCUDA_EventTming()
{
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );
}

float stopCUDA_EventTiming()
{
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );

	float elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );

	return elapsedTime;
}

//===============================================================================
// 运行V1并行脸部检测内核
//==============================================================================

std::vector<CvRect> runGPUHaarDetection(std::vector<double> scale, int minNeighbors, FaceDetectionKernel kernelSelect)
{
	printf("****Beginning GPU(Kernel %d) Haar Detection****\n\n", kernelSelect + 1);

	std::vector<CvRect> faces;
	float totalElapsedTime = 0.0f;
	for(int i = 0; i < scale.size(); i++)
	{
		// 修改新尺度下的所有特征
		h_gpuHaarCascade.setFeaturesForScale(scale[i]);
		
		// 拷贝新尺度下的值到设备中
		size_t GPU_Classifier_Size = h_gpuHaarCascade.totalNumOfClassifiers * sizeof(GPUHaarClassifier);
		HANDLE_ERROR( cudaMemcpy(dev_gpuHaarCascade.scaled_haar_classifiers, h_gpuHaarCascade.scaled_haar_classifiers, GPU_Classifier_Size, cudaMemcpyHostToDevice));

		dev_gpuHaarCascade.scale = h_gpuHaarCascade.scale;
		dev_gpuHaarCascade.real_window_size = h_gpuHaarCascade.real_window_size;
		dev_gpuHaarCascade.img_detection_size = h_gpuHaarCascade.img_detection_size;

		int w = dev_gpuHaarCascade.img_detection_size.width;
		int h = dev_gpuHaarCascade.img_detection_size.height;

		// 基于输入选择，启动相应的内核
		float elapsedTime;
		switch(kernelSelect)
		{
			case V1:
				elapsedTime = launchKernel_v1(w,  h);
				break;
			case V2:
				elapsedTime = launchKernel_v2(w,  h);
				break;
			case V3:
				elapsedTime = launchKernel_v3(w, h);
				break;
			case V4:
				elapsedTime = launchKernel_v4(w,  h);
				break;
		}

		totalElapsedTime += elapsedTime;

		// 从设备中拷贝数据并在CPU中进行处理
		HANDLE_ERROR( cudaMemcpy(detectedFaces, dev_detectedFaces, detectedFacesSize, cudaMemcpyDeviceToHost));
	
		// 扫描detectedFaces数组获取检测到的脸部区域信息
		int faces_detected = selectFaces(faces, detectedFaces, w * h);

		// 输出耗时信息等
		printf("Stage: %d // Faces Detected: %d // GPU Time: %3.1f ms \n", i, faces_detected, elapsedTime);
	}

	// 输出最终表现
	printf("\nTotal compute time: %3.1f ms \n\n", totalElapsedTime);

	// 将检测到的脸部进行分组，更好表现结果
	if( minNeighbors != 0)
	{
		groupRectangles(faces, minNeighbors, GROUP_EPS);
	}

	// 清空数组，用于后续处理
	memset(detectedFaces, 0, detectedFacesSize);
	HANDLE_ERROR( cudaMemcpy(dev_detectedFaces, detectedFaces, detectedFacesSize, cudaMemcpyHostToDevice));

	return faces;
}

// 对于图像中每个检测窗口，调用GPU内核来运行haar cascade
float launchKernel_v1(int width, int height)
{
	// 定义块的数目和线程数目来分配工作量
	dim3    blocks(width/16, height/16);
	dim3    threads(16, 16);

	startCUDA_EventTming();
	haarDetection_v1<<<blocks, threads>>>(dev_gpuHaarCascade, dev_detectedFaces);
	return stopCUDA_EventTiming(); 
}

float launchKernel_v2(int width, int height)
{
	dim3    blocks(width/16, height/16);
	dim3    threads(16, 16);

	startCUDA_EventTming();
	haarDetection_v2<<<blocks, threads>>>(dev_gpuHaarCascade, dev_detectedFaces);
	return stopCUDA_EventTiming(); 
}

float launchKernel_v3(int width, int height)
{
	dim3	blocks(32);
	dim3    threads(8, 8);

	startCUDA_EventTming();
	haarDetection_v3<<<blocks, threads>>>(dev_gpuHaarCascade, dev_detectedFaces);
	return stopCUDA_EventTiming(); 
}

float launchKernel_v4(int width, int height)
{
	int size = sqrt((float)THREADS_PER_BLOCK_V4);

	dim3    blocks(width/size, height/size);
	dim3    threads(size, size);

	// 对于这个内核需要对每个块进行初始lock操作，后续优化可以考虑使用其他方式的mutex locking

	int numOfLocks = blocks.x * blocks.y;
	Lock *h_locks = (Lock *)malloc(numOfLocks * sizeof(Lock));

	// 初始化locks中的mutex变量
	for(int i = 0; i < numOfLocks; i++)
		h_locks[i].init();

	Lock * dev_locks;
	HANDLE_ERROR( cudaMalloc( (void**)&dev_locks, numOfLocks * sizeof(Lock) ) );
	HANDLE_ERROR( cudaMemcpy(dev_locks, h_locks,  numOfLocks * sizeof(Lock), cudaMemcpyHostToDevice));

	startCUDA_EventTming();
	haarDetection_v4<<<blocks, threads>>>(dev_gpuHaarCascade, dev_detectedFaces, dev_locks);
	float time = stopCUDA_EventTiming(); 

	free(h_locks);
	HANDLE_ERROR(cudaFree(dev_locks));

	return time;
}

// unbind纹理并释放内存（host以及device）
void shutDownGPU()
{
	releaseTextures();

	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );

	h_gpuHaarCascade.shutdown();
	HANDLE_ERROR( cudaFree(dev_gpuHaarCascade.haar_classifiers));
	HANDLE_ERROR( cudaFree(dev_gpuHaarCascade.scaled_haar_classifiers));

	free(detectedFaces);
	HANDLE_ERROR( cudaFree(dev_detectedFaces));

	HANDLE_ERROR( cudaFreeArray(dev_sumArray));
	HANDLE_ERROR( cudaFreeArray(dev_sqSumArray));

	HANDLE_ERROR( cudaDeviceReset());
}