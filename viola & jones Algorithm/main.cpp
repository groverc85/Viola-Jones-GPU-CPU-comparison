#include <stdio.h>
#include <stdlib.h>
#include "image.h"
#include "stdio-wrapper.h"
#include "haar.h"

#define INPUT_FILENAME "1000*536.pgm"
#define OUTPUT_FILENAME "1000*536_vj.pgm"

using namespace std;


int main (int argc, char *argv[]) 
{

	int flag;
	
	int mode = 1;
	int i;

	/* detection parameters */
	float scaleFactor = 1.2;
	int minNeighbours = 1;


	printf("-- entering main function --\r\n");

	printf("-- loading image --\r\n");

	MyImage imageObj;
	MyImage *image = &imageObj;

	flag = readPgm((char *)"1000*536.pgm", image);
	if (flag == -1)
	{
		printf( "Unable to open input image\n");
		return 1;
	}

	printf("-- loading cascade classifier --\r\n");

	myCascade cascadeObj;
	myCascade *cascade = &cascadeObj;
	MySize minSize = {20, 20};
	MySize maxSize = {0, 0};

	/* classifier properties */
	cascade->n_stages=25;
	cascade->total_nodes=2913;
	cascade->orig_window_size.height = 24;
	cascade->orig_window_size.width = 24;


	readTextClassifier();

	std::vector<MyRect> result;

	printf("-- detecting faces --\r\n");

	result = detectObjects(image, minSize, maxSize, cascade, scaleFactor, minNeighbours);

	for(i = 0; i < result.size(); i++ )
	{
		MyRect r = result[i];
		drawRectangle(image, r);
	}

	printf("-- saving output --\r\n"); 
	flag = writePgm((char *)OUTPUT_FILENAME, image); 

	printf("-- image saved --\r\n");

	/* delete image and free classifier */
	releaseTextClassifier();
	freeImage(image);

	return 0;
}
