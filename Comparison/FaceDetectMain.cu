 #include "FaceDetectExtras.h"
#include "GPUHaarCascade.h"
#include "GPU_Face_Detect.cuh"
#include "CPU_Face_Detect.h"
#include "OpenCV_Face_Detect.h"

using namespace std;
using namespace cv;

static void CheckError()
{
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		printf("CUDA error1: %s\n", cudaGetErrorString(error));
		system("pause");
	}
}

// 遍历OpenCV Cascade来判断该cascade下classifiers的数目
int numberOfClassifiers(CvHaarClassifierCascade *cvCascade)
{
	int totalClassifiers = 0;
	for(int i = 0; i < cvCascade->count; i++)
	{
		CvHaarStageClassifier stage = cvCascade->stage_classifier[i];

		totalClassifiers += stage.count;
	}

	return totalClassifiers;
}

void displayResults(IplImage * image, std::vector<CvRect> faces, char * windowTitle)
{
	// 创建矩形来展示结果
	for(int i = 0; i < faces.size(); i++)
	{
		CvRect face_rect = faces[i];

		cvRectangle( image, 
				cvPoint(face_rect.x, face_rect.y),
				cvPoint((face_rect.x + face_rect.width), (face_rect.y + face_rect.height)),
				CV_RGB(255, 255, 255), 3);
	}
	
	cvNamedWindow( windowTitle, 0 );
	cvShowImage( windowTitle, image );
	//cvWaitKey(0);
}

IplImage * createCopy(IplImage *img)
{
	IplImage *cpyImg = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels); 
	cvCopy(img, cpyImg);
	return cpyImg;
}

int main( int argc, const char** argv )
{
	//load测试图片和OpenCVHaarCascade XML文件进行检测

    int optlen = strlen("--cascade=");

	const char *imageFileName = "images\\lena_256.jpg";
	const char *detectorFileName;

    // 检测输入格式是否正确
    if( argc > 1 && strncmp( argv[1], "--cascade=", optlen ) == 0 )
    {
        detectorFileName = argv[1] + optlen;
        imageFileName = argc > 2 ? argv[2] : imageFileName;
    }
    else
	{
		printf("Incorrect input for command line. Using default values instead.\n");
		printf("Correct Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n\n" );

		detectorFileName = "data\\haarcascade_frontalface_default.xml";
    }

	// 装载图像
	IplImage* image;
	if((image = cvLoadImage(imageFileName, CV_LOAD_IMAGE_GRAYSCALE)) == 0)	
	{
		cout << "Error occured loading image file. Check file name?" << endl;
		system("pause");
		return 0;
	}

	int width = image->width;
	int height = image->height;
	CvSize imgSize = cvSize(width, height);

	printf("Input image: %s\n", imageFileName);
	printf("Image size: [%d, %d]\n\n", width, height);

	//load OpenCVHaarCascade并创建system GPUHaarCascade
	CvHaarClassifierCascade *cvCascade = loadCVHaarCascade(detectorFileName);
	GPUHaarCascade gpuHaarCascade;

	//将OpenCV Haar Cascade数据结构转为GPU Haar Cascade结构
	gpuHaarCascade.load(cvCascade, numberOfClassifiers(cvCascade), imgSize); 

	printf("Input Detector: %s\n", detectorFileName);
	printf("Num of Stages: %d\n", gpuHaarCascade.numOfStages);
	printf("Num of Classifiers: %d\n\n", gpuHaarCascade.totalNumOfClassifiers);

	//积分图像计算
	CvMat* sum = cvCreateMat(height + 1, width + 1, CV_32SC1);
	CvMat* sqsum = cvCreateMat(height + 1, width + 1, CV_64FC1);

	cvIntegral(image, sum, sqsum);

	//计算放缩值，在每次检测之后重新调整检测窗口大小
	double factor = 1.0f;
	float scaleFactor = 1.2f;

	std::vector<double> scale;
	
	while(factor * gpuHaarCascade.orig_window_size.width < width - 10 &&
		  factor * gpuHaarCascade.orig_window_size.height < height - 10)
	{
		scale.push_back(factor);
		factor *= scaleFactor;
	}
	
	// 用于分组检测到的矩形（即脸部区域）
	int minNeighbors = 3;

	// 运行GPU脸部检测
	initGPU(gpuHaarCascade, image, sum, sqsum);
	
	IplImage *gpuImage_v1 = createCopy(image);
	std::vector<CvRect> gpuFaces_v1 = runGPUHaarDetection(scale, minNeighbors, V1);
	
	IplImage *gpuImage_v3 = createCopy(image);
	std::vector<CvRect> gpuFaces_v3;// = runGPUHaarDetection(scale, minNeighbors, V3);
	
	IplImage *gpuImage_v4 = createCopy(image);
	std::vector<CvRect> gpuFaces_v4;// = runGPUHaarDetection(scale, minNeighbors, V4);
	
	shutDownGPU();

	// 运行CPU脸部检测
	Mat sum_Mat = cvarrToMat(sum);
	Mat sqsum_Mat = cvarrToMat(sqsum);

	IplImage *cpuImage = createCopy(image);
	std::vector<CvRect> cpuFaces = runCPUHaarDetection(gpuHaarCascade, imgSize, sum_Mat, sqsum_Mat, scale, minNeighbors);

	IplImage *cpuImage_Multithread = createCopy(image);
	runCPUHaarDetection_Multithread(gpuHaarCascade, imgSize, sum_Mat, sqsum_Mat, scale, minNeighbors);

	// 运行OpenCV脸部检测
	IplImage *opencvImage = createCopy(image);
	std::vector<CvRect> opencvFaces = runOpenCVHaarDetection(image, cvCascade, scaleFactor);

	// 结果展示
	displayResults(gpuImage_v1, gpuFaces_v1, "GPU Results v1");
	displayResults(gpuImage_v3, gpuFaces_v3, "GPU Results v3");
	displayResults(gpuImage_v4, gpuFaces_v4, "GPU Results v4");

	displayResults(cpuImage, cpuFaces, "CPU Results");
	displayResults(cpuImage_Multithread, gpuFaces_v3, "CPU_Multithread Results");

	displayResults(opencvImage, opencvFaces, "OpenCV Results");
	
	cvWaitKey(0);
	system("pause");

	//释放内存
	cvReleaseHaarClassifierCascade( &cvCascade );

	cvReleaseImage(&image);
	cvReleaseMat(&sum);
	cvReleaseMat(&sqsum);

	cvReleaseImage(&image);
	cvReleaseImage(&gpuImage_v1);
	cvReleaseImage(&gpuImage_v3);
	cvReleaseImage(&gpuImage_v4);
	cvReleaseImage(&cpuImage);
	cvReleaseImage(&cpuImage_Multithread);
	cvReleaseImage(&opencvImage);

	gpuHaarCascade.shutdown();

	return 0;
}


