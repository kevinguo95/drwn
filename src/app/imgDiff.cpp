/*imgDiff
A program to compare two images

*/

#pragma warning(disable: 4996)

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
//#include <iomanip>


// opencv library headers
#include "cv.h"
#include "highgui.h"

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnML.h"
#include "drwnPGM.h"
#include "drwnVision.h"


using namespace std;
using namespace Eigen;
using namespace cv;


void usage() {
	cout << "segmented image   ground truth   output file"
	<< endl;

}

// main ----------------------------------------------------------------------

int main(int argc, char *argv[])
{
	cout << "Hello World";
	double matchPcent = 0;
	DRWN_ASSERT(argc == 4);
		
	const char *segFile = argv[1];
	const char *groundFile = argv[2];
	const char *outputName = argv[3];
	const char *colorFile = "../tests/input/images/21077.jpg";
	ofstream output;
	output.open(outputName);
	drwnGrabCutInstance model;

	cv::Mat img = imread(colorFile, CV_LOAD_IMAGE_COLOR);
	cv::Mat segImg = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat groundImg = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	
	DRWN_ASSERT(img.data != NULL);
	DRWN_ASSERT(segImg.data != NULL);
	DRWN_ASSERT(groundImg.data != NULL);

	model.initialize(img, segImg, groundImg, NULL);
	matchPcent = model.loss(segImg);

	output << matchPcent;
	output << "testttt";
	output.close();

	cout << matchPcent << "ajllsfjgsjkfkjjsd";
	return EXIT_SUCCESS;
}