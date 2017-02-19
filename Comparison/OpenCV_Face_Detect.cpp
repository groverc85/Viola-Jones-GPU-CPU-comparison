#include "OpenCV_Face_Detect.h"

using namespace std;
using namespace cv;

CvHaarClassifierCascade* load_object_detector( const char* cascade_path )
{
    return (CvHaarClassifierCascade*)cvLoad( cascade_path );
}

std::vector<CvRect> detectObjects( IplImage* image, CvHaarClassifierCascade* cascade, int do_pyramids, float scaleFactor )
{
    IplImage* small_image = image;
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* faces;

    // 进行image pyramid操作
    if(do_pyramids)
    {
        small_image = cvCreateImage( cvSize(image->width/2,image->height/2), IPL_DEPTH_8U, 3 );
        cvPyrDown( image, small_image, CV_GAUSSIAN_5x5 );
        scaleFactor = 2;
    }

    /* use the fastest variant */
    faces = cvHaarDetectObjects( small_image, cascade, storage, scaleFactor, 2, CV_HAAR_DO_CANNY_PRUNING );

	std::vector<CvRect> returnFaces;	
	for(int i = 0; i < faces->total; i++ )
	{
		// Extract the rectangles only
		CvRect face_rect = *(CvRect*)cvGetSeqElem( faces, i );
		returnFaces.push_back(face_rect);
	}

    if( small_image != image )
        cvReleaseImage( &small_image );
    cvReleaseMemStorage( &storage );

	return returnFaces;
}

std::vector<CvRect> runOpenCVHaarDetection(IplImage *image, CvHaarClassifierCascade* cascade, float scaleFactor)
{
	printf("****Beginning OpenCV Haar Detection****\n\n");
	clock_t start, end;

	std::vector<CvRect> outputFaces;

	start = clock();

		outputFaces = detectObjects( image, cascade, 0, scaleFactor);

	end = clock();
	double elapsedTime = (double)end - start;
	printf("\nTotal compute time: %3.1f ms \n\n", elapsedTime);

	return outputFaces;
}