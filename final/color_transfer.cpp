#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstring>
#define SEP_NUM 1 //will split src_img into n*n rectangles
#define VERSION 2

using namespace cv;
using namespace std;

class LabData {
public:
	double l, a, b;
	LabData() {l = 0, a = 0, b = 0;}
	void sqrt_data() {l = sqrt(l), a = sqrt(a), b = sqrt(b);}
	void print() {printf("l:%lf, a:%lf, b:%lf\n", l, a, b);}
	double& get(int i) {return (i == 0) ? l : ((i == 1) ? a : b);}
};

void get_means(Mat &mat, LabData &mean)
{
	int pixels = mat.rows * mat.cols;
	for(int y = 0; y < mat.rows; y++) {
		unsigned char* data = mat.ptr<uchar>(y);
		for(int x = 0; x < mat.cols; x++) {
			for(int i = 0; i < 3; i++) {
				mean.get(i) += (double)(*data) / pixels;
				data++;
			}
		}
	}
}

void get_sigma(Mat &mat, LabData &mean, LabData &sigma)
{
	int pixels = mat.rows * mat.cols;
	for(int y = 0; y < mat.rows; y++) {
		unsigned char* data = mat.ptr<uchar>(y);
		for(int x = 0; x < mat.cols; x++) {
			for(int i = 0; i < 3; i++) {
				sigma.get(i) += pow(((double)(*data) - mean.get(i)), 2) / pixels;
				data++;
			}
		}
	}
	sigma.sqrt_data();
}

void fix_mean_and_sigma(Mat &mat, LabData &old_mean, LabData &old_sigma) //Ignore the extreme values
{
	int pixels = mat.rows * mat.cols;
	printf("pixes:%d\n", pixels);
	int count[3] = {0};
	LabData new_mean;
	for(int y = 0; y < mat.rows; y++) {
		unsigned char* data = mat.ptr<uchar>(y);
		for(int x = 0; x < mat.cols; x++) {
			for(int i = 0; i < 3; i++) {
				if( abs((double)(*data)-old_mean.get(i)) < old_sigma.get(i)*1.5) {
					new_mean.get(i) += (double)(*data) / pixels;
					count[i]++;
				}
				data++;
			}
		}
	}
	for(int i = 0; i < 3; i++)
		new_mean.get(i) *= ((double)pixels / count[i]);
	LabData new_sigma;
	for(int y = 0; y < mat.rows; y++) {
		unsigned char* data = mat.ptr<uchar>(y);
		for(int x = 0; x < mat.cols; x++) {
			for(int i = 0; i < 3; i++) {
				if( abs((double)(*data)-old_mean.get(i)) < old_sigma.get(i)*1.5)
					new_sigma.get(i) += pow(((double)(*data) - new_mean.get(i)), 2) / count[i];
				data++;
			}
		}
	}
	new_sigma.sqrt_data();
	for(int i = 0; i < 3; i++) {
		old_mean.get(i) = new_mean.get(i);
		old_sigma.get(i) = new_sigma.get(i);
		printf("count%d : %d\n", i, count[i]);
	}
	
}

Mat* split_mat(Mat &src)
{
	Mat *mats = new Mat[SEP_NUM * SEP_NUM];
	int width = src.cols / SEP_NUM;
	int hight = src.rows / SEP_NUM;
	for(int i = 0; i < SEP_NUM; i++) {
		for(int j = 0; j < SEP_NUM; j++) {
			mats[i*SEP_NUM+j] = src(Rect(j*width, i*hight, width, hight));
		}
	}
	return mats;
}

void get_index(int &index, int x, int y, Mat &mat)
{
	int Y = (y / (mat.rows/SEP_NUM) ) * SEP_NUM;
	if(Y > SEP_NUM * (SEP_NUM-1) )
		Y = SEP_NUM * (SEP_NUM-1);
	int X = x / (mat.cols/SEP_NUM);
	if(X >= SEP_NUM)
		X = SEP_NUM - 1;
	index = Y + X;
}

void scale(Mat &mat)
{
	if(mat.rows > 600) {
		double ratio = 600 / (double)mat.rows;
		resize(mat, mat, Size(), ratio, ratio, CV_INTER_AREA);
	}
	if(mat.cols > 800) {
		double ratio = 800 / (double)mat.cols;
		resize(mat, mat, Size(), ratio, ratio, CV_INTER_AREA);
	}
}

double distance(int x, int y, Point &center)
{
	double dis = sqrt(pow(center.x - x, 2) + pow(center.y - y, 2) );
	if(dis < 1) dis = 1;
	return dis;
}
int main(int argc, char** argv)
{
	if(argc != 3) {
		cout << "Input format error" << endl;
		return -1;
	}
	Mat src = imread(argv[1] , CV_LOAD_IMAGE_UNCHANGED);
	if(src.empty() ) {
		cout << "Cannot open source image." << endl;
		return -1;
	}
	Mat tar = imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);
	if(tar.empty() ) {
		cout << "Cannot open target image." << endl;
		return -1;
	}
	Mat src_lab;
	cvtColor(src, src_lab ,CV_BGR2Lab);
	Mat tar_lab;
	cvtColor(tar, tar_lab, CV_BGR2Lab);
	vector<LabData> mean_src;
	vector<LabData> sigma_src;
	LabData mean_tar, sigma_tar;
	
	Mat *src_labs = split_mat(src_lab);
	Point center[SEP_NUM*SEP_NUM];
	int width = src_lab.cols/2, height = src_lab.rows/2;
	for(int i = 0; i < SEP_NUM*SEP_NUM; i++) {
		mean_src.push_back(LabData());
		sigma_src.push_back(LabData());
		get_means(src_labs[i], mean_src.at(i));
		get_sigma(src_labs[i], mean_src.at(i), sigma_src.at(i));
		center[i] = Point( (int)(i%SEP_NUM+0.5)*width, (int)(i/SEP_NUM+0.5)*height);
	}
	
	LabData all_mean_src, all_sigma_src;
	get_means(src_lab, all_mean_src);
	get_sigma(src_lab, all_mean_src, all_sigma_src);
	printf("all_mean_src: %lf, %lf, %lf\n", all_mean_src.l, all_mean_src.a, all_mean_src.b);
	printf("all_sigma_src: %lf, %lf, %lf\n", all_sigma_src.l, all_sigma_src.a, all_sigma_src.b);
	
	get_means(tar_lab, mean_tar);
	get_sigma(tar_lab, mean_tar, sigma_tar);
	printf("old mean_tar: %lf, %lf, %lf\n", mean_tar.l, mean_tar.a, mean_tar.b);
	printf("old sigma_tar: %lf, %lf, %lf\n", sigma_tar.l, sigma_tar.a, sigma_tar.b);
	fix_mean_and_sigma(tar_lab, mean_tar, sigma_tar);
	/*for(int i = 0; i < SEP_NUM*SEP_NUM; i++) {
		printf("src[%d]: ", i);
		mean_src.at(i).print();
		sigma_src.at(i).print();
	}*/
	printf("new mean_tar: %lf, %lf, %lf\n", mean_tar.l, mean_tar.a, mean_tar.b);
	printf("new sigma_tar: %lf, %lf, %lf\n", sigma_tar.l, sigma_tar.a, sigma_tar.b);

	int counter[3] = {0};
	for(int y = 0; y < src_lab.rows; y++) {
		for(int x = 0; x < src_lab.cols; x++) {
			Vec3b color_src = src_lab.at<Vec3b>(Point(x, y));
			int index;
			get_index(index, x, y, src_lab);
			for(int i = 0; i < 3; i++) {
				double ratio = sigma_tar.get(i)/all_sigma_src.get(i);
				double val_old = (color_src.val[i]-all_mean_src.get(i)) * ratio + mean_tar.get(i);
				double val_new = (color_src.val[i]-mean_src.at(index).get(i)) * ratio + mean_tar.get(i);
				double dis_weight = 1 / sqrt(distance(x, y, center[index]));
				double val = val_new * dis_weight + val_old * (1 - dis_weight);
				if(val > 255)
					val = 255;
				else if(val < 0)
					val = 0;
				else if(abs(color_src.val[i] - val) > abs(color_src.val[i]-mean_tar.get(i))*3 
						) {
					color_src.val[i] = val * 0.7 + color_src.val[i] * 0.3;
					counter[i]++;
				}
				else
					color_src.val[i] = val;
			}
			src_lab.at<Vec3b>(Point(x, y)) = color_src;
		}
	}
	for(int i = 0; i < 3; i++)
		printf("counter[%d]: %d", i, counter[i]);
	printf("\n");
	Mat res;
	cvtColor(src_lab, res, CV_Lab2BGR);
	/*scale(src);
	scale(tar);
	scale(res);*/
	stringstream ss;
	ss << "img/res_" << VERSION << "." << SEP_NUM << ".jpg";
	string name = ss.str();
	//imwrite(name, res);
	scale(src);
	scale(tar);
	scale(res);
	namedWindow("src", WINDOW_AUTOSIZE );
	imshow("src", src);
	namedWindow("tar", WINDOW_AUTOSIZE );
	imshow("tar", tar);
	namedWindow("res", WINDOW_AUTOSIZE );
	imshow("res", res);
	waitKey(0);
	return 0;
}