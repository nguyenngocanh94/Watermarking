#include <opencv2/opencv.hpp>
#include <istream>
#include <iostream>
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
/**************************************functions**********************************************/
//from vector to Mat
template<typename T=uchar>
inline cv::Mat idecoupage(const std::vector<cv::Mat_<T>>& vBlocks, const cv::Size& oImageSize, int nChannels) {
	std::vector<cv::Mat_<T>> mergedMatrix(nChannels);
	cv::Mat outputImage;
	int nbBlocbycols = oImageSize.width / 8; // nombre de bloc par colonne
	int nbBlocbyrows = oImageSize.height / 8; // nombre de bloc par ligne

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
				mt2.push_back(vBlocks[c*nbBlocbycols *nbBlocbyrows + i*nbBlocbycols + j]);		// Ajout du bloc dans le vecteur	
			}
			cv::hconcat(mt2,t2); // Concatenation de la matrice avec le vecteur (Ajout par colonne)
			mt1.push_back(t2); // Ajout du vecteur dans la matrice
			
		}
		cv::vconcat(mt1, t1); // Concatenation de la matrice avec le vecteur (Ajout par ligne)
		mergedMatrix[c].push_back(t1); // Ajout du vecteur dans la matrice finale
	}
	cv::merge(mergedMatrix, outputImage); // Transforme en matrice RGB
	return outputImage;
}
//from Mat to vector
template<typename T=uchar>
inline std::vector<cv::Mat_<T>> decoupage(const cv::Mat& oImage) 
{
	CV_Assert(oImage.depth()==CV_8U);

	int nbBlocbycols = oImage.cols/ 8; // nombre de bloc par colonne
	int nbBlocbyrows = oImage.rows / 8;// nombre de bloc par ligne
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
//zigzag scan from mat to vector
template<int nBlockSize = 8>
inline std::vector<int> zigzag(const cv::Mat_<float>& mat) 
{
    CV_Assert(!mat.empty());
    CV_Assert(mat.rows==mat.cols && mat.rows==nBlockSize);
	int nIdx = 0;
	std::vector<int> zigzag(nBlockSize*nBlockSize);
	
	for (int i = 0; i < nBlockSize * 2; ++i)
		for (int j = (i < nBlockSize) ? 0 : i - nBlockSize + 1; j <= i && j < nBlockSize; ++j)
			zigzag[nIdx++] = mat((i & 1) ? j*(nBlockSize - 1) + i : (i - j)*nBlockSize + j);

	return zigzag;
}
//izigzag from vector to mat
template<int nBlockSize=8,typename T=float>
inline cv::Mat_<T> izigzag(const std::vector<T>& vec) 
{
    CV_Assert(!vec.empty());
    CV_Assert(int(vec.size())==nBlockSize*nBlockSize);
    int nIdx = 0;
    cv::Mat_<T> oMatResult(nBlockSize*nBlockSize,1);
    for(int i=0; i<nBlockSize*2; ++i)
        for(int j=(i<nBlockSize) ? 0 : i-nBlockSize+1; j<=i && j<nBlockSize; ++j)
            oMatResult((i&1) ? j*(nBlockSize-1)+i :(i-j)*nBlockSize+j) = vec[nIdx++];
    return oMatResult.reshape(0,nBlockSize);
}
//generate a ramdom map
int myrandom (int i) { return std::rand()%i;}
inline vector<int> generator(int a,int b)
{
  ofstream myfile ("example.txt");
  srand ( unsigned ( std::time(0) ) );
  vector<int> myvector;
  for (int i=0; i<a*b; ++i) myvector.push_back(i);
  random_shuffle ( myvector.begin(), myvector.end() );
  random_shuffle ( myvector.begin(), myvector.end(), myrandom);
  if (myfile.is_open())
  {
    for(int i=0; i<myvector.size(); ++i)
    {
      myfile << myvector[i] << ' ';
    }
    myfile.close();
  }
  return myvector;
}
// calculate SMSE, PSNR
float calRms(Mat originIMG, Mat wmIMG)
{
	float e;
	unsigned a = 0;
	int m = originIMG.rows, n = originIMG.cols;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a = a + pow((wmIMG.at<uchar>(i, j) - originIMG.at<uchar>(i, j)), 2);
		}
	}
	e = pow(a / m / n, 0.5);
	return e;
}

float calSnr(Mat originIMG, Mat wmIMG)
{
	float s;
	unsigned a = 0, b = 0;
	int m = originIMG.rows, n = originIMG.cols;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a = a + pow((wmIMG.at<uchar>(i, j) - originIMG.at<uchar>(i, j)), 2);
			b = b + pow(wmIMG.at<uchar>(i, j), 2);
		}
	}
	s = b / a;
	return s;
}
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
/***************************************process watermark********************************************************/
// devide to 8*8 block, next scrambler the watermark image, then binarizied it
vector<int> convertTo_binary(Mat img){
	vector<int> convert;
	int blocks = img.cols/8;
	vector<Mat_<uchar>> vectorBlock = decoupage(img);
	Mat create(8,8,CV_8UC1,Scalar::all(0));
	vector<Mat_<uchar>> scramvectorBlock(blocks*blocks,create);
	vector<int> generate = generator(blocks,blocks);
	for (int i = 0; i < blocks*blocks; ++i)
	{
		int index = generate[i];
		scramvectorBlock[i] = vectorBlock[index];
	}
	Mat wmIMG = idecoupage(scramvectorBlock,img.size(), img.channels());
	namedWindow("after",WINDOW_AUTOSIZE);
	imshow("after",wmIMG);
	waitKey(0);
	for (int i = 0; i < img.rows; ++i)
	{	
		for (int j = 0; j < img.cols; ++j)
		{
			if (img.at<uchar>(i,j) > 100)
			{
				convert.push_back(1);
			}else
			convert.push_back(0);
		}
	}
	return convert;
}
/*******************************process Image ****************************************************/
//function ProcessImageBeforeWMembeded : divide to vector of block, then dct, quantum, at least. zigzag to transform to vector of vector
vector<vector<int> > ProcessImageBeforeWMembeded(Mat OriginImage){
	vector<Mat_<uchar>> vectorBlock = decoupage(OriginImage);
	vector<vector<int> > vectorAfterDCT;
	for (int i = 0; i < vectorBlock.size(); ++i)
	{
		//read 1 block
		Mat img(8, 8, CV_32FC1);
		for (int r = 0; r < 8; ++r)
		{
			for (int p = 0; p < 8; ++p)
			{
				img.at<float>(r, p) = vectorBlock[i].at<uchar>(r, p);
			}
		}
		dct(img-128, img); //dct
		//quantum
		Mat_<int> Matquantum(8,8,CV_32FC1);
		Mat Q = (Mat_<float>(8,8)<<  3,2,2,3,5,8,10,12,2,2,3,4,5,12,12,11,3,3,3,5,8,11,14,11,3,3,4,6,10,17,16,12,4,4,7,11,14,22,21,15,5,7,11,13,16,12,23,18,10,13,16,17,21,24,24,21,14,18,19,20,22,20,20,20);				
		for (int i = 0; i < 8; ++i)
		{	
			for (int j = 0; j < 8; ++j)
			{
				Matquantum.at<int>(i,j) = round(img.at<float>(i,j)/Q.at<float>(i,j));
			}
		}
		//zigzag then pushback to vector<vector>
		vector<int> v;
		v = zigzag(Matquantum);
		vectorAfterDCT.push_back(v);
	}
	return vectorAfterDCT; //type int
}
/*********************************************re - construct ********************************************/
// izigzag the vector of vector,dequantum, idct and transform to vector of Mat
vector<Mat_<float>> reconstruct(vector<vector<float> > vectorConvert, int blocks){
	vector<Mat_<float>> vectorContruct(vectorConvert.size(),Mat(8,8,CV_32FC1));
	Mat Q = (Mat_<float>(8,8)<<  3,2,2,3,5,8,10,12,2,2,3,4,5,12,12,11,3,3,3,5,8,11,14,11,3,3,4,6,10,17,16,12,4,4,7,11,14,22,21,15,5,7,11,13,16,12,23,18,10,13,16,17,21,24,24,21,14,18,19,20,22,20,20,20);
	for (int i = 0; i < blocks; ++i)
	{	
		Mat temp(8,8,CV_32FC1);
		vectorContruct[i] = izigzag(vectorConvert[i]);
	}
	return vectorContruct;
}
/******************************************** MAIN ********************************************/
int main(int argc, char const *argv[])
{
	Mat OriginImg;
	float watermark_scalor = 10.5;
	OriginImg = imread("lena.jpg",0);
	Mat imgcopy = OriginImg.clone();
	int blocks = (OriginImg.rows/8)*(OriginImg.cols/8);
	Mat Watermark;
	Watermark = imread("BKHN.jpg",0);
	vector<int> vectorWM = convertTo_binary(Watermark);
	vector<vector<int> > vectorAfterDCT = ProcessImageBeforeWMembeded(OriginImg);
	vector<vector<float> > vectorConvert(vectorAfterDCT.size(),vector<float>(64));
	for (int i = 0; i < vectorAfterDCT.size(); ++i)
	{
		for (int j = 0; j < 64; ++j)
		{
			vectorConvert[i].at(j) = vectorAfterDCT[i].at(j);
		}
	}
	for (int m = 0,flag = 0; flag < vectorWM.size() ; m++, flag++)
	{
		if (m<blocks)
		{
			if (vectorWM[m]==1)
			{
				if (vectorConvert.at(m).at(6+2*(flag/blocks)) == vectorConvert.at(m).at(6+1+2*(flag/blocks)))
				{
					vectorConvert.at(m).at(6+1+2*(flag/blocks)) = vectorConvert.at(m).at(6+1+2*(flag/blocks)) + watermark_scalor; // watermark_scalor
				}
				if (vectorConvert.at(m).at(6+2*(flag/blocks)) > vectorConvert.at(m).at(6+1+2*(flag/blocks)))
				{
					swap(vectorConvert.at(m).at(6+2*(flag/blocks)),vectorConvert.at(m).at(6+1+2*(flag/blocks)));
				}
			}else{
				if (vectorConvert.at(m).at(6+2*(flag/blocks)) == vectorConvert.at(m).at(6+1+2*(flag/blocks)))
				{
					vectorConvert.at(m).at(6+2*(flag/blocks)) = vectorConvert.at(m).at(6+2*(flag/blocks)) + watermark_scalor; // watermark_scalor
				}	
				if (vectorConvert.at(m).at(6+2*(flag/blocks)) < vectorConvert.at(m).at(6+1+2*(flag/blocks)))
				{
					swap(vectorConvert.at(m).at(6+2*(flag/blocks)),vectorConvert.at(m).at(6+1+2*(flag/blocks)));
				}
			}
		}else{
			if (vectorWM[m]==1)
			{
				if (vectorConvert.at(m).at(6+2*(flag/blocks)) == vectorConvert.at(m).at(6+1+2*(flag/blocks)))
				{
					vectorConvert.at(m).at(6+1+2*(flag/blocks)) += watermark_scalor; // watermark_scalor
				}
				if (vectorConvert.at(m).at(6+2*(flag/blocks)) > vectorConvert.at(m).at(6+1+2*(flag/blocks)))
				{
					swap(vectorConvert.at(m).at(6+2*(flag/blocks)),vectorConvert.at(m).at(6+1+2*(flag/blocks)));
				}
			}else{
				if (vectorConvert.at(m).at(6+2*(flag/blocks)) == vectorConvert.at(m).at(6+1+2*(flag/blocks)))
				{
					vectorConvert.at(m).at(6+2*(flag/blocks)) += watermark_scalor; // watermark_scalor
				}
				if (vectorConvert.at(m).at(6+2*(flag/blocks)) < vectorConvert.at(m).at(6+1+2*(flag/blocks)))
				{
					swap(vectorConvert.at(m).at(6+2*(flag/blocks)),vectorConvert.at(m).at(6+1+2*(flag/blocks)));
				}
			}
		}
	}
	Mat Q = (Mat_<float>(8,8)<<  3,2,2,3,5,8,10,12,2,2,3,4,5,12,12,11,3,3,3,5,8,11,14,11,3,3,4,6,10,17,16,12,4,4,7,11,14,22,21,15,5,7,11,13,16,12,23,18,10,13,16,17,21,24,24,21,14,18,19,20,22,20,20,20);
    vector<Mat_<float>> vectorContruct(vectorConvert.size(),Mat(8,8,CV_32FC1));
	vector<Mat_<uchar>> vectorImg(vectorConvert.size(),Mat(8,8,CV_8UC1));
	vector<Mat_<float>> vectordequantum(vectorConvert.size(),Mat(8,8,CV_32FC1));
	vectorContruct = reconstruct(vectorConvert, blocks);
	Mat_<float> temp(8,8,CV_32FC1);
	for (int i = 0; i < vectorContruct.size();i++)
	{	
		for (int k = 0; k < 8; ++k)
		{
			for (int l = 0; l < 8; ++l)
			{
				temp.at<float>(k,l) = vectorContruct[i].at<float>(k,l) * Q.at<float>(k,l);;
			}
		}
		cout << temp << endl;
	}
	for (int i = 0; i < 4096; ++i)
	{
		cout << "Mat dequantum = " << endl << " " << vectordequantum[i] << endl << endl;
	}
	for (int i = 0; i < blocks; ++i)
	{
		for (int j = 0; j < 8; ++j)
		{
			for (int k = 0; k < 8; ++k)
			{
				vectorImg[i].at<uchar>(j,k) = vectordequantum[i].at<float>(j,k) + 128;
			}
		}
	}
	Mat remapImg;
	remapImg = idecoupage(vectorImg,OriginImg.size(),OriginImg.channels());
	for (int u = 0; u < imgcopy.rows; u++)
	{
		for (int v = 0; v < imgcopy.cols; v++)
		{
			imgcopy.at<uchar>(u, v) = remapImg.at<uchar>(u, v);
		}
	}
	namedWindow("source image",WINDOW_AUTOSIZE);
	namedWindow("transform",WINDOW_AUTOSIZE);
	imshow("source image", OriginImg);
	imshow("transform", imgcopy);
	waitKey(0);
	return 0;
}