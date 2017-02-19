#include "opencv\cv.h"
#ifndef GPUHAARCASCADE_H
#define GPUHAARCASCADE_H

typedef struct GPURect
{
	float x;
	float y;
	float width;
	float height;
#ifdef __cplusplus
	__device__ __host__
	GPURect(float _x = 0, float _y = 0, float w = 0, float h = 0) : x(_x), y(_y), width(w), height(h) {}
#endif
}GPURect;

typedef struct GPUFeatureRect
{
	GPURect r;
	float weight;
}GPUFeatureRect;

typedef struct GPUHaarFeature
{
	GPUFeatureRect rect0;
	GPUFeatureRect rect1;
	GPUFeatureRect rect2;
}GPUHaarFeature;

typedef struct GPUHaarClassifier
{
	//假定每个classifier只有一个feature
	GPUHaarFeature haar_feature;
	float threshold;

	//对应alpha[]
	float alpha0;
	float alpha1;
}GPUHaarClassifier;

typedef struct GPUHaarStageClassifier
{
	//该阶段classifiers的数目
	int  numofClassifiers; 
	//classifiers的阈值
    float threshold; 
    //index下标指向该阶段的classifiers开始
	int classifierOffset;
}GPUHaarStageClassifier;

typedef struct GPUHaarCascade
{
	// CvHaarClassifierCascade参数
	int  flags; /* signature */
    int  numOfStages; /* number of stages */
	int totalNumOfClassifiers; /* cascade中的classfier数目 */
    CvSize orig_window_size; /* cascade训练时候的object size */
	CvSize img_window_size; /* 原始窗口大小 */

	CvSize img_detection_size; /* 在haar检测运行时设定 */

    /* 通过cvSetImagesForHaarClassifierCascade进行设定 */
    CvSize real_window_size; /* current object size */
    double scale; /* 当前尺度 */

	// 数组存储当前系统内所有stage classifiers
    GPUHaarStageClassifier* haar_stage_classifiers; 

	// 数组存储cascade内所有classifiers
	GPUHaarClassifier * haar_classifiers;

	// 数组存储超过尺度大小的修改后的所有classifiers
	GPUHaarClassifier * scaled_haar_classifiers;
	
	void load(CvHaarClassifierCascade *cvCascade, int clasifiersCount, CvSize imgSize)
	{
		// 传递一些基本的cascade参数（先前设定）
		flags = cvCascade->flags;
		numOfStages = cvCascade->count;
		orig_window_size = cvCascade->orig_window_size;
		real_window_size = cvCascade->real_window_size;
		img_window_size = imgSize;
		scale = cvCascade->scale;
		totalNumOfClassifiers = clasifiersCount;

		haar_stage_classifiers = (GPUHaarStageClassifier *)malloc(numOfStages * sizeof(GPUHaarStageClassifier));
		haar_classifiers = (GPUHaarClassifier *)malloc(totalNumOfClassifiers * sizeof(GPUHaarClassifier));
		scaled_haar_classifiers = (GPUHaarClassifier *)malloc(totalNumOfClassifiers * sizeof(GPUHaarClassifier));

		// 遍历 OpenCV Cascade tree来传递Stage_Classifiers & Classifiers data
		int gpuClassifierCounter = 0;
		for(int i = 0; i < cvCascade->count; i++)
		{
			// 使用OpenCV stage classifier创建GPU Stage Classifier
			CvHaarStageClassifier stage = cvCascade->stage_classifier[i];
		
			GPUHaarStageClassifier gpuStage;
			gpuStage.threshold = stage.threshold;
			gpuStage.numofClassifiers = stage.count;
			gpuStage.classifierOffset = gpuClassifierCounter;

			// 遍历当前阶段的所有classifiers
			for(int j = 0; j < stage.count; j++)
			{
				CvHaarClassifier classifier = stage.classifier[j];

				if(classifier.count > 1)
				{
					// 抛出错误
					printf("Can't handle HaarFeature xml files with classifiers with more than 1 haar feature.\n");
					return;
				}

				// 创建GPU Classifier
				GPUHaarClassifier gpuClassifier;
				gpuClassifier.threshold = classifier.threshold[0];
				gpuClassifier.alpha0 = classifier.alpha[0];
				gpuClassifier.alpha1 = classifier.alpha[1];

				// 使用OpenCV stage classifier创建GPU Stage Classifier
				CvHaarFeature feature = classifier.haar_feature[0];

				GPUHaarFeature gpuFeature;
				gpuFeature.rect0.r.x = feature.rect[0].r.x;
				gpuFeature.rect0.r.y = feature.rect[0].r.y;
				gpuFeature.rect0.r.width = feature.rect[0].r.width;
				gpuFeature.rect0.r.height = feature.rect[0].r.height;
				gpuFeature.rect0.weight = feature.rect[0].weight;

				gpuFeature.rect1.r.x = feature.rect[1].r.x;
				gpuFeature.rect1.r.y = feature.rect[1].r.y;
				gpuFeature.rect1.r.width = feature.rect[1].r.width;
				gpuFeature.rect1.r.height = feature.rect[1].r.height;
				gpuFeature.rect1.weight = feature.rect[1].weight;

				gpuFeature.rect2.r.x = feature.rect[2].r.x;
				gpuFeature.rect2.r.y = feature.rect[2].r.y;
				gpuFeature.rect2.r.width = feature.rect[2].r.width;
				gpuFeature.rect2.r.height = feature.rect[2].r.height;
				gpuFeature.rect2.weight = feature.rect[2].weight;

				gpuClassifier.haar_feature = gpuFeature;

				// 增加新的GPU Classifier到GPUCascade中的classifiers数组中
				haar_classifiers[gpuClassifierCounter] = gpuClassifier;
				scaled_haar_classifiers[gpuClassifierCounter] = gpuClassifier;
				gpuClassifierCounter++;
			}

			// 增加新的GPU stage classifier到数组中
			haar_stage_classifiers[i] = gpuStage;
		}
	}

	void load(GPUHaarCascade * gpuCascade)
	{
		// 传递一些基本的cascade参数（先前设定）
		flags = gpuCascade->flags;
		numOfStages = gpuCascade->numOfStages;
		orig_window_size = gpuCascade->orig_window_size;
		real_window_size = gpuCascade->real_window_size;
		img_window_size = gpuCascade->img_window_size;
		scale = gpuCascade->scale;
		totalNumOfClassifiers = gpuCascade->totalNumOfClassifiers;

		// 分配拷贝stage clasifiers
		haar_stage_classifiers = (GPUHaarStageClassifier *)malloc(numOfStages * sizeof(GPUHaarStageClassifier));
		for(int i = 0; i < numOfStages; i++)
			haar_stage_classifiers[i] = gpuCascade->haar_stage_classifiers[i];

		// 分配拷贝clasifiers
		haar_classifiers = (GPUHaarClassifier *)malloc(totalNumOfClassifiers * sizeof(GPUHaarClassifier));
		scaled_haar_classifiers = (GPUHaarClassifier *)malloc(totalNumOfClassifiers * sizeof(GPUHaarClassifier));
		for(int i = 0; i < totalNumOfClassifiers; i++)
		{
			haar_classifiers[i] = gpuCascade->haar_classifiers[i];
			scaled_haar_classifiers[i] = gpuCascade->scaled_haar_classifiers[i];
		}
	}

	void setFeaturesForScale(float newScale)
	{
		scale = newScale;
		real_window_size.width = cvRound(orig_window_size.width * scale);
		real_window_size.height = cvRound(orig_window_size.height * scale);

		img_detection_size.width = cvRound(img_window_size.width - real_window_size.width);
		img_detection_size.height = cvRound(img_window_size.height - real_window_size.height);

		for(int i = 0; i < totalNumOfClassifiers; i++)
		{
			GPUHaarFeature original_feature = haar_classifiers[i].haar_feature;
			GPUHaarFeature *scaled_feature = &scaled_haar_classifiers[i].haar_feature;

			scaled_feature->rect0.r.x = original_feature.rect0.r.x * scale;
			scaled_feature->rect0.r.y = original_feature.rect0.r.y * scale;
			scaled_feature->rect0.r.width = original_feature.rect0.r.width * scale;
			scaled_feature->rect0.r.height = original_feature.rect0.r.height * scale;
			
			scaled_feature->rect1.r.x = original_feature.rect1.r.x * scale;
			scaled_feature->rect1.r.y = original_feature.rect1.r.y * scale;
			scaled_feature->rect1.r.width = original_feature.rect1.r.width * scale;
			scaled_feature->rect1.r.height = original_feature.rect1.r.height * scale;
			
			if(original_feature.rect2.weight)
			{
				scaled_feature->rect2.r.x = original_feature.rect2.r.x * scale;
				scaled_feature->rect2.r.y = original_feature.rect2.r.y * scale;
				scaled_feature->rect2.r.width = original_feature.rect2.r.width * scale;
				scaled_feature->rect2.r.height = original_feature.rect2.r.height * scale;
			}
		}
	}

	void shutdown()
	{
		free(haar_stage_classifiers);
		free(haar_classifiers);
	}

}GPUHaarCascade;

#endif