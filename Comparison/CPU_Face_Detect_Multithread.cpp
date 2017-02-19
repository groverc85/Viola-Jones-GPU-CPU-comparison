#include "CPU_Face_Detect.h"

#include <Windows.h>

using namespace std;
using namespace cv;

Mat * sumImg;
Mat * sqSumImg;

float calculateMean_Multithread(CvRect rect)
{
	return sumImg->at<int>(rect.y,rect.x) 
			- sumImg->at<int>(rect.y,rect.x + rect.width ) 
			- sumImg->at<int>(rect.y + rect.height , rect.x) 
			+ sumImg->at<int>(rect.y + rect.height ,rect.x + rect.width);
}

float calculateSum_Multithread(GPURect rect, int win_start_x, int win_start_y, float scale)
{
	float tx = win_start_x + rect.x * scale;
	float ty = win_start_y + rect.y * scale;

	return sumImg->at<int>(ty,tx) 
			- sumImg->at<int>(ty,tx + rect.width * scale) 
			- sumImg->at<int>(ty + rect.height * scale, tx) 
			+ sumImg->at<int>(ty + rect.height * scale, tx + rect.width * scale);
}

struct cpu_thread_data
{
	int id;
	int width;
	int height;

	float elapsedTime;
	
	GPUHaarCascade gpuCascade;
};

DWORD WINAPI haarDetection(LPVOID lpParameter)
{
	cpu_thread_data *data = (cpu_thread_data *)lpParameter;

	clock_t startCPU = clock();

	GPUHaarCascade gpuCascade = data->gpuCascade;
	int faces_detected = 0;
	for(int i = 0; i < data->width; i++)
	{
		for(int j = 0; j < data->height; j++)
		{
			CvRect detectionWindow;
			detectionWindow.x = i;
			detectionWindow.y = j;
			detectionWindow.width = gpuCascade.real_window_size.width;
			detectionWindow.height = gpuCascade.real_window_size.height;

			float inv_window_area = 1.0f / ((float)detectionWindow.width * detectionWindow.height);
			float weightScale = inv_window_area;

			// HaarCascade需要对特征进行normalization
			float mean = calculateMean_Multithread(detectionWindow);

			float variance_norm_factor = sqSumImg->at<double>(detectionWindow.y,detectionWindow.x) 
				- sqSumImg->at<double>(detectionWindow.y,detectionWindow.x + detectionWindow.width) 
				- sqSumImg->at<double>(detectionWindow.y + detectionWindow.height, detectionWindow.x) 
				+ sqSumImg->at<double>(detectionWindow.y + detectionWindow.height, detectionWindow.x + detectionWindow.width);
		
			variance_norm_factor = variance_norm_factor * inv_window_area - mean * mean;
			if(variance_norm_factor >= 0.0f)
				variance_norm_factor = sqrt(variance_norm_factor);
			else
				variance_norm_factor = 1.0f;

			bool passed = true;

			for(int a = 0; a < gpuCascade.numOfStages; a++)
			{
				float stage_sum = 0.0;
				for(int b = 0; b < gpuCascade.haar_stage_classifiers[a].numofClassifiers; b++)
				{
					int index = b +  gpuCascade.haar_stage_classifiers[a].classifierOffset;
					GPUHaarClassifier classifier = gpuCascade.haar_classifiers[index];
					
					double t = classifier.threshold * variance_norm_factor;

					double sum = calculateSum_Multithread(classifier.haar_feature.rect0.r, detectionWindow.x, detectionWindow.y, gpuCascade.scale) * classifier.haar_feature.rect0.weight * weightScale;
					sum += calculateSum_Multithread(classifier.haar_feature.rect1.r, detectionWindow.x, detectionWindow.y, gpuCascade.scale) * classifier.haar_feature.rect1.weight * weightScale;

					if(classifier.haar_feature.rect2.weight)
						sum += calculateSum_Multithread(classifier.haar_feature.rect2.r, detectionWindow.x, detectionWindow.y, gpuCascade.scale) * classifier.haar_feature.rect2.weight * weightScale;
            
					if(sum >= t)
						stage_sum += classifier.alpha1;
					else
						stage_sum += classifier.alpha0;
				}

				// 若Classifier没有通过，则排除整个cascade
				if( stage_sum < gpuCascade.haar_stage_classifiers[a].threshold)
				{
					passed = false;
					break;
				}
			}
	
			if(passed)
				faces_detected++;
		}
	}

	clock_t endCPU = clock();
	double totalElapsedTime = (double)endCPU - startCPU;
	data->elapsedTime = totalElapsedTime;

	printf("Stage: %d // Faces Detected: %d // GPU Time: %3.1f ms \n", data->id, faces_detected, totalElapsedTime);

	return 0;
}

std::vector<CvRect> runCPUHaarDetection_Multithread(GPUHaarCascade & cascade, CvSize imgSize, Mat sum, Mat sqsum, std::vector<double> scale, int minNeighbors)
{
	printf("****Beginning Multi-Threaded CPU Haar Detection****\n\n");

	// 拷贝到本地变量
	sumImg = &sum;
	sqSumImg = &sqsum;

	std::vector<CvRect> allDetectedFaces;

	int maxThreads = scale.size();
	cpu_thread_data * threadData = (cpu_thread_data *)malloc(maxThreads * sizeof(cpu_thread_data));
	HANDLE  * hThreadArray = (HANDLE *)malloc(maxThreads * sizeof(HANDLE));

	for(int i = 0; i < maxThreads; i++)
	{
		threadData[i].id = i;
		threadData[i].gpuCascade = cascade;

		threadData[i].gpuCascade.scale = scale[i];
		threadData[i].gpuCascade.real_window_size.width = cvRound(cascade.orig_window_size.width * scale[i]);
		threadData[i].gpuCascade.real_window_size.height = cvRound(cascade.orig_window_size.height * scale[i]);

		int w = cvRound(imgSize.width - threadData[i].gpuCascade.real_window_size.width);
		int h = cvRound(imgSize.height - threadData[i].gpuCascade.real_window_size.height);

		threadData[i].width = w;
		threadData[i].height = h;

		// 创建thread来运行haar检测
		hThreadArray[i] = CreateThread( NULL, 0, haarDetection, (void*)&threadData[i], 0, 0);

		if (hThreadArray[i] == NULL) 
		{
			printf("Error occured while trying to create thread");
			ExitProcess(3);
		}
	}

	// 等待直到所有线程都结束
	WaitForMultipleObjects(maxThreads, hThreadArray, TRUE, INFINITE);
	
	float totalElapsedTime = 0.0f;
	for(int i = 0; i < maxThreads; i++)
		totalElapsedTime += threadData[i].elapsedTime;

	// 输出最终表现
	printf("\nTotal compute time: %3.1f ms \n\n", totalElapsedTime);

	// 将检测到的脸部进行分组，更好表现结果
	if( minNeighbors != 0)
	{
		groupRectangles(allDetectedFaces, minNeighbors, GROUP_EPS);
	}

	// 关闭所有线程handle
	for(int i = 0; i < maxThreads; i++)
	{
		CloseHandle(hThreadArray[i]);
	}

	// 释放内存
	free(threadData);
	free(hThreadArray);

	return allDetectedFaces;
}
