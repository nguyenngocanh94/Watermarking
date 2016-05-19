#include <opencv2/opencv.hpp>
#include <istream>
#include <stdio.h>
#include <iterator>
#include <string>
#include <vector>
#include <math.h>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <algorithm>
#include <fstream>

using namespace std;
using namespace cv;
int abc(int a){
	return a;
}
int main(int argc, char const *argv[])
{	
	Mat white(64,64,CV_8UC1,Scalar::all(255));
	namedWindow("white",WINDOW_AUTOSIZE);
	imshow("white",white);
	waitKey(0);
	return 0;
}