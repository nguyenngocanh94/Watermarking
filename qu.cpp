#include <opencv2/opencv.hpp>
#include <istream>
#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include <algorithm>
#include <fstream>

using namespace std;
using namespace cv;
int main(int argc, char const *argv[])
{	
	Mat quantum = (Mat_<float>(8,8) << 26,-3,-6,2,2,-1,0,-2,-4,1,1,0,0,0,-3,1,5,-1,-1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
	Mat img(8,8,CV_32FC1,Scalar::all(0));
	Mat Q = (Mat_<float>(8,8) << 16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99);
	for (int i = 0; i < 8; ++i)
	{
		for (int j = 0;j < 8; ++j)
		{
			img.at<float>(i,j) = Q.at<float>(i,j) * quantum.at<float>(i,j);
		}
	}
	cout << img << endl;
	return 0;
}