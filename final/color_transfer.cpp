#include "opencv2/opencv.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#define SEP_NUM 1 //will split src_img into n*n rectangles

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

void get_means_sigma_fix(Mat &mat, LabData &mean, LabData &sigma)// This will ignore the extreme values
{
	int pixels = mat.rows * mat.cols;
	double *data_l = new double[pixels];
	double *data_a = new double[pixels];
	double *data_b = new double[pixels];
	for(int y = 0; y < mat.rows; y++) {
		unsigned char* data = mat.ptr<uchar>(y);
		for(int x = 0; x < mat.cols; x++) {
			data_l[y*(mat.cols)+x] = (double)(*data);
			data_a[y*(mat.cols)+x] = (double)(*(data+1));
			data_b[y*(mat.cols)+x] = (double)(*(data+2));
			data += 3;
		}
	}
	sort(data_l, data_l+pixels);
	sort(data_a, data_a+pixels);
	sort(data_b, data_b+pixels);
	int num = pixels * 0.8;
	for(int i = pixels * 0.1; i < pixels * 0.9; i++) {
		mean.get(0) += data_l[i] / num;
		mean.get(1) += data_a[i] / num;
		mean.get(2) += data_b[i] / num;
	}
	for(int i = pixels * 0.1; i < pixels * 0.9; i++) {
		sigma.get(0) += pow(data_l[i] - mean.get(0), 2) / num;
		sigma.get(1) += pow(data_a[i] - mean.get(1), 2) / num;
		sigma.get(2) += pow(data_b[i] - mean.get(2), 2) / num;
	}
	sigma.sqrt_data();
	delete []data_l;
	delete []data_a;
	delete []data_b;
}

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

void scale_img(Mat &mat)
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

void scale_ratio(double &ratio)
{
	if(ratio > 5) {
		ratio = 5 + sqrt(ratio);
	} else  {
		while (ratio < 0.7)
			ratio = sqrt(ratio);
	}
}

double distance(int x, int y, Point &center)
{
	double dis = sqrt(pow(center.x - x, 2) + pow(center.y - y, 2) );
	if(dis < 5)
		dis = 5;
	return dis;
}
int main(int argc, char** argv)
{
	if(argc < 3) {
		fprintf(stderr, "Input format error.\n");
		return -1;
	}
	Mat src = imread(argv[1] , CV_LOAD_IMAGE_UNCHANGED);
	if(src.empty() ) {
		fprintf(stderr, "Cannot open source image.\n");
		return -1;
	}
	Mat tar = imread(argv[2], CV_LOAD_IMAGE_UNCHANGED);
	if(tar.empty() ) {
		fprintf(stderr, "Cannot open target image.\n");
		return -1;
	}
	int color_space = 0;
	Mat src_new;
	cvtColor(src, src_new ,CV_BGR2Lab);
	Mat tar_new;
	cvtColor(tar, tar_new, CV_BGR2Lab);
	if(argc > 3) {
		if(strcmp(argv[3], "Luv") == 0) {
			cvtColor(src, src_new ,CV_BGR2Luv);
			cvtColor(tar, tar_new, CV_BGR2Luv);
			color_space = 1;
		} else if(strcmp(argv[3], "HSV") == 0) {
			cvtColor(src, src_new ,CV_BGR2HSV);
			cvtColor(tar, tar_new, CV_BGR2HSV);
			color_space = 2;
		} else if(strcmp(argv[3], "HLS") == 0) {
			cvtColor(src, src_new ,CV_BGR2HSV);
			cvtColor(tar, tar_new, CV_BGR2HSV);
			color_space = 3;
		}
	}
	
	vector<LabData> mean_src;
	vector<LabData> sigma_src;
	Mat *src_news = split_mat(src_new);
	Point center[SEP_NUM*SEP_NUM];
	int width = src_new.cols/SEP_NUM, height = src_new.rows/SEP_NUM;
	for(int i = 0; i < SEP_NUM*SEP_NUM; i++) {
		mean_src.push_back(LabData());
		sigma_src.push_back(LabData());
		get_means(src_news[i], mean_src.at(i));
		get_sigma(src_news[i], mean_src.at(i), sigma_src.at(i));
		center[i] = Point( (int)((i%SEP_NUM+0.5)*width), (int)((i/SEP_NUM+0.5)*height));
	}
	LabData all_mean_src, all_sigma_src;
	get_means(src_new, all_mean_src);
	get_sigma(src_new, all_mean_src, all_sigma_src);
	printf("all_mean_src: %lf, %lf, %lf\n", all_mean_src.l, all_mean_src.a, all_mean_src.b);
	printf("all_sigma_src: %lf, %lf, %lf\n", all_sigma_src.l, all_sigma_src.a, all_sigma_src.b);
	
	LabData mean_tar, sigma_tar;
	get_means_sigma_fix(tar_new, mean_tar, sigma_tar);
	printf("mean_tar: %lf, %lf, %lf\n", mean_tar.l, mean_tar.a, mean_tar.b);
	printf("sigma_tar: %lf, %lf, %lf\n", sigma_tar.l, sigma_tar.a, sigma_tar.b);

	for(int y = 0; y < src_new.rows; y++) {
		for(int x = 0; x < src_new.cols; x++) {
			Vec3b color_src = src_new.at<Vec3b>(Point(x, y));
			Vec3b color_left = src_new.at<Vec3b>(Point((x>0)?x-1:0, y));
			Vec3b color_up = src_new.at<Vec3b>(Point(x, (y>0)?y-1:0));
			int index;
			get_index(index, x, y, src_new);
			for(int i = 0; i < 3; i++) {
				double ratio = sigma_tar.get(i)/all_sigma_src.get(i);
				scale_ratio(ratio);
				double val_old = (color_src.val[i]-all_mean_src.get(i)) * ratio + mean_tar.get(i);
				double val_new = (color_src.val[i]-mean_src.at(index).get(i)) * ratio + mean_tar.get(i);
				double dis_weight = 1 / distance(x, y, center[index]);
				double val = val_new * dis_weight + val_old * (1 - dis_weight);
				if(val > 255 || val < 0) {
					if(x > 0 && y > 0)
						val = (color_left.val[i] + color_up.val[i]) / 2;
					else
						val = (val > 255) ? 255 : 0;
				}
				if(abs(color_src.val[i] - val) > abs(color_src.val[i]-mean_tar.get(i))*3)
					color_src.val[i] = val * 0.6 + (color_left.val[i] + color_up.val[i]) * 0.2;
				else
					color_src.val[i] = val;
			}
			src_new.at<Vec3b>(Point(x, y)) = color_src;
		}
	}
	Mat res;
	switch(color_space) {
		case 0:
			cvtColor(src_new, res, CV_Lab2BGR);
			break;
		case 1:
			cvtColor(src_new, res, CV_Luv2BGR);
			break;
		case 2:
			cvtColor(src_new, res, CV_HSV2BGR);
			break;
		case 3:
			cvtColor(src_new, res, CV_HLS2BGR);
			break;
	}
	char name[30];
	sprintf(name, "img/res_%s_%s.jpg", (argc>3)?argv[3]:"lab", strtok(argv[1], "."));
	imwrite(name, res);
	scale_img(src);
	scale_img(tar);
	scale_img(res);
	namedWindow("src", WINDOW_AUTOSIZE);
	imshow("src", src);
	namedWindow("tar", WINDOW_AUTOSIZE);
	imshow("tar", tar);
	namedWindow("res", WINDOW_AUTOSIZE);
	imshow("res", res);
	waitKey(0);
	return 0;
}