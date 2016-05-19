#include <opencv2/opencv.hpp>
#include <istream>
#include <stdio.h>
#include <iterator>
#include <string>
#include <vector>
#include <math.h>
#include <ctime>         //std::time
#include <cstdlib>       //std::rand, std::srand
#include <algorithm>
#include <fstream>

using namespace std;
using namespace cv;
/**************************************functions**********************************************/
template<typename T=uchar>
inline cv::Mat idecoupage(const std::vector<cv::Mat_<T>>& vBlocks, const cv::Size& oImageSize, int nChannels) {
	std::vector<cv::Mat_<T>> mergedMatrix(nChannels);
	cv::Mat outputImage;
	int nbBlocbycols = oImageSize.width / 8;
	int nbBlocbyrows = oImageSize.height / 8;

	for (int c = 0; c<nChannels; ++c)
	{
		cv::Mat t1;
		std::vector<cv::Mat> mt1;
		for (int i = 0; i<nbBlocbyrows; ++i) 
		{
			cv::Mat t2;
			std::vector<cv::Mat> mt2;
			for (int j = 0; j<nbBlocbycols; ++j) 
			{
				mt2.push_back(vBlocks[c*nbBlocbycols *nbBlocbyrows + i*nbBlocbycols + j]);	
			}
			cv::hconcat(mt2,t2);
			mt1.push_back(t2);
			
		}
		cv::vconcat(mt1, t1); 
		mergedMatrix[c].push_back(t1); 
	}
	cv::merge(mergedMatrix, outputImage); 
	return outputImage;
}
//from Mat to vector
template<typename T=uchar>
inline std::vector<cv::Mat_<T>> decoupage(const cv::Mat& oImage) 
{
	CV_Assert(oImage.depth()==CV_8U);

	int nbBlocbycols = oImage.cols/ 8; 
	int nbBlocbyrows = oImage.rows / 8;
	std::vector<cv::Mat_<T>> blocMat(nbBlocbycols * nbBlocbyrows *oImage.channels());
	std::vector<cv::Mat_<T>> rgbChannels(oImage.channels());
	
	if (oImage.channels() == 1)
	{
		rgbChannels[0] = oImage;
	}
	else
	{
		cv::split(oImage, rgbChannels);
	}

	for (int c = 0; c < rgbChannels.size(); ++c)
	{
		for (int i = 0; i<nbBlocbyrows; ++i) 
		{
			for (int j = 0; j<nbBlocbycols; ++j) 
			{
				blocMat[c*nbBlocbyrows *nbBlocbycols + i * nbBlocbycols + j] = rgbChannels[c].cv::Mat::colRange(j * 8, (j + 1) * 8).cv::Mat::rowRange(i * 8, (i + 1) * 8);
			}
		}
	}

	return blocMat;
}
//function get map pseu-random
vector<int> getmap(){
	ifstream is("example.txt");
  	istream_iterator<int> start(is), end;
  	vector<int> numbers(start, end);
  	return numbers;
}
//function check watermark
int checkwatermark(vector<float> v, int k){
	if (v[6+2*k] < v[7+2*k])
	{	
		return 1;
	}else{
		return 0;
	}
}
//zigzag
template<int nBlockSize = 8>
inline std::vector<float> zigzag(const cv::Mat_<float>& mat) 
{
    CV_Assert(!mat.empty());
    CV_Assert(mat.rows==mat.cols && mat.rows==nBlockSize);
	int nIdx = 0;
	std::vector<float> zigzag(nBlockSize*nBlockSize);
	
	for (int i = 0; i < nBlockSize * 2; ++i)
		for (int j = (i < nBlockSize) ? 0 : i - nBlockSize + 1; j <= i && j < nBlockSize; ++j)
			zigzag[nIdx++] = mat((i & 1) ? j*(nBlockSize - 1) + i : (i - j)*nBlockSize + j);

	return zigzag;
}
/*******************get index of ***************************************************/
// find index of value in map
int indexof(vector<int> A,int a){
	for (int i = 0; i < A.size(); ++i)
	{
		if (A[i]==a)
		{
			return i;
			break;
		}
	}
}
/*************************************************MAIN**********************************************/
int main(int argc, char const *argv[])
{
	Mat originIMG = imread("watermark.jpg", 0);
	if (originIMG.empty())
	{
		cout << "error!" << endl;
		return -1;
	}
	int a = originIMG.rows /8 +1;
	int b = originIMG.cols /8 +1;
	int blocks = (a-1)*(b-1);
	Mat imgcopy = originIMG.clone();
	vector<int> scramble;
	scramble = getmap();
	Mat background(scramble.size(),scramble.size(),CV_8UC1,Scalar::all(255));
	int watermarkbits = scramble.size() * 64;
	vector<int> vectorwm;

	for (int i = 0,flag = 0;flag < watermarkbits; i++)
	{
		for (int j = 0; j <b-1; j++)
		{
			Mat img(8, 8, CV_32FC1);
			//read 1 block
			if (i<a-1)
			{
				for (int r = i * 8, f = 0; r < i * 8 + 8, f < 8; r++, f++)
				{
					for (int t = j * 8, g = 0; t < j * 8 + 8, g < 8; t++, g++)
					{
						img.at<float>(f, g) = imgcopy.at<uchar>(r, t);
					}
				}
			}else{
				for (int r = (i-a-1) * 8, f = 0; r < (i-a-1) * 8 + 8, f < 8; r++, f++)
				{
					for (int t = j * 8, g = 0; t < j * 8 + 8, g < 8; t++, g++)
					{
						img.at<float>(f, g) = imgcopy.at<uchar>(r, t);
					}
				}	
			}
			dct(img-128, img); //dct
			//zigzag scan 
			vector<float> v1;
			v1 = zigzag(img);
			int a = checkwatermark(v1,flag/blocks);
			vectorwm.push_back(a);	
			flag++;
		}
	}
/************************** contruct the image ****************************/
	Mat watermarkimg(64,64,CV_8UC1,Scalar::all(0));
	for (int i = 0, k = 0; i < scramble.size(); ++i)
	{
		for (int j = 0; j < scramble.size(); ++j)
		{
			if (vectorwm[k] == 0)
			{	
				watermarkimg.at<uchar>(i,j) = 0;

			}else{
				watermarkimg.at<uchar>(i,j) = 255;
			}
			k++;
		}
	}	
/*****************************load watermarking on background************************************************/
	for (int i = 0; i < 64; ++i)
	{
		for (int j = 0; j < 64 ; ++j)
		{
			background.at<uchar>(i,j) = watermarkimg.at<uchar>(i,j);
		}
	}
/************************************* de-scramble ********************************************/
	vector<Mat_<uchar>> scramblevectorBlock = decoupage(background);
	Mat create(8,8,CV_8UC1,Scalar::all(0));
	vector<Mat_<uchar>> vectorBlock(scramble.size(),create);
	for (int i = 0; i < scramble.size(); ++i)
	{
		int index = indexof(scramble,i);
		vectorBlock[i] = scramblevectorBlock[index];
	}
	Mat wmIMG = idecoupage(vectorBlock,background.size(), background.channels());
	/**********************/	
	namedWindow("watermark_img",WINDOW_AUTOSIZE);
	imshow("watermark_img",wmIMG);
	imwrite("watermark-extract.jpg",wmIMG);	
	waitKey(0);
	return 0;
}
