#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <iostream>
#include <ctime>
#include <windows.h>
#include<omp.h>
using namespace cv;
int colorList[4][3] = { 0 };

int computeChangeColor(double* pixel, int colorList[][3]) {
	int edgeColor[8][3] = { { 0,0,0 },{ 255,255,255 },{ 255,0,0 },{ 0,255,0 },{ 0,0,255 }, { 255,255,0 },{ 0,255,255 },{ 255,0,255 } };
	std::vector<double> D(4);
	std::vector<double> Drgb(3);
	std::vector<double> De(8);
	long long sum = 0;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 3; j++)
			Drgb[j] = abs(pixel[j] - colorList[i][j]);
		sum = std::accumulate(Drgb.begin(), Drgb.end(), 0);
		for (int j = 0; j < 8; j++)
			De[j] = abs(colorList[i][0] - edgeColor[j][0]) + abs(colorList[i][1] - edgeColor[j][1]) + abs(colorList[i][2] - edgeColor[j][2]);
		sort(De.begin(), De.end());
		D[i] = sum + De[1];
	}
	return std::distance(D.begin(), std::min_element(D.begin(), D.end()));
}

void KmeansImage(Mat src) {
	src.convertTo(src, CV_64FC3);
	double prev_colorList[4][3] = { 0 };
	std::vector<std::vector<int>> colormap(src.size().height, std::vector<int>(src.size().width));
	srand(time(NULL));
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 3; j++) {
			colorList[i][j] = rand() % 256;
		}
	}
	int N = 3;
	while (N) {
		for (int i = 0; i < src.size().height; i++) {
			for (int j = 0; j < src.size().width; j++) {
				double* pixel = src.ptr<double>(i, j);
				colormap[i][j] = computeChangeColor(pixel, colorList);
			}
		}

		std::vector<double> colorNum(4);
		std::vector<std::vector<double>> colorSum(4, std::vector<double>(3));
		for (int i = 0; i < src.size().height; i++) {
			for (int j = 0; j < src.size().width; j++) {
				int colorGroup = colormap[i][j];
				double* pixel = src.ptr<double>(i, j);
				colorNum[colorGroup]++;
				colorSum[colorGroup][0] += pixel[0];
				colorSum[colorGroup][1] += pixel[1];
				colorSum[colorGroup][2] += pixel[2];
			}
		}

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 3; j++) {
				if (colorNum[i] != 0) {
					colorList[i][j] = colorSum[i][j] / colorNum[i];
				}
				else {
					colorList[i][j] = rand() % 256;
				}
			}
		}

		int equ = 1;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 3; j++) {
				if (abs(prev_colorList[i][j] - colorList[i][j]) > 2) {
					equ = 0;
					break;
				}
			}
		}

		if (equ)N--;

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 3; j++) {
				prev_colorList[i][j] = colorList[i][j];
			}
		}
	}
}

void KmeansImage_parallel(Mat src) {
	src.convertTo(src, CV_64FC3);
	double prev_colorList[4][3] = { 0 };
	std::vector<std::vector<int>> colormap(src.size().height, std::vector<int>(src.size().width));
	srand(time(NULL));
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 3; j++) {
			colorList[i][j] = rand() % 256;
		}
	}
	int N = 3;
	while (N) {
		omp_set_num_threads(4);
		#pragma omp parallel for
		for (int i = 0; i < src.size().height; i++) {
			//omp_set_num_threads(4);
			#pragma omp parallel for
			for (int j = 0; j < src.size().width; j++) {
				double* pixel = src.ptr<double>(i, j);
				colormap[i][j] = computeChangeColor(pixel, colorList);
			}
		}

		std::vector<double> colorNum(4);
		std::vector<std::vector<double>> colorSum(4, std::vector<double>(3));
		for (int i = 0; i < src.size().height; i++) {
			for (int j = 0; j < src.size().width; j++) {
				int colorGroup = colormap[i][j];
				double* pixel = src.ptr<double>(i, j);
				colorNum[colorGroup]++;
				colorSum[colorGroup][0] += pixel[0];
				colorSum[colorGroup][1] += pixel[1];
				colorSum[colorGroup][2] += pixel[2];
			}
		}

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 3; j++) {
				if (colorNum[i] != 0) {
					colorList[i][j] = colorSum[i][j] / colorNum[i];
				}
				else {
					colorList[i][j] = rand() % 256;
				}
			}
		}

		int equ = 1;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 3; j++) {
				if (abs(prev_colorList[i][j] - colorList[i][j]) > 2) {
					equ = 0;
					break;
				}
			}
		}

		if (equ)N--;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 3; j++) {
				prev_colorList[i][j] = colorList[i][j];
			}
		}
	}
}

Mat dithering(Mat src) {
	Mat dst(src.size(), CV_8UC3);
	src.convertTo(src, CV_64FC3);
	for (int i = 0; i < src.size().height; i++) {
		for (int j = 0; j < src.size().width; j++) {
			double* pixel = src.ptr<double>(i, j);
			int n = computeChangeColor(pixel, colorList);
			std::vector<double>error(3);
			for (int k = 0; k < 3; k++) {
				error[k] = pixel[k] - colorList[n][k];
				dst.ptr(i, j)[k] = colorList[n][k];
			}

			if (j != src.size().width - 1) {
				for (int k = 0; k < 3; k++)
					src.ptr<double>(i, j + 1)[k] += error[k] * 6 / 17.0;
			}
			if (i != src.size().height - 1) {
				if (j != 0) {
					for (int k = 0; k < 3; k++)
						src.ptr<double>(i + 1, j - 1)[k] += ((error[k] * 3 / 17.0));
				}
				for (int k = 0; k < 3; k++)
					src.ptr<double>(i + 1, j)[k] += error[k] * 5 / 17.0;
				if (j != src.size().width - 1) {
					for (int k = 0; k < 3; k++)
						src.ptr<double>(i + 1, j + 1)[k] += error[k] * 1 / 17.0;
				}
			}
		}
	}
	return dst;
}

void processPiexl(Mat& src, Mat& dst, int x, int y) {
	double* pixel = src.ptr<double>(x, y);
	int n = computeChangeColor(pixel, colorList);
	std::vector<double>error(3);
	for (int k = 0; k < 3; k++) {
		error[k] = pixel[k] - colorList[n][k];
		dst.ptr(x, y)[k] = colorList[n][k];
		if (y != src.size().width - 1) {
			src.ptr<double>(x, y + 1)[k] += error[k] * 6 / 17.0;
		}
		if (x != src.size().height - 1) {
			if (y != 0) {
				src.ptr<double>(x + 1, y - 1)[k] += ((error[k] * 3 / 17.0));
			}
			src.ptr<double>(x + 1, y)[k] += error[k] * 5 / 17.0;
			if (y != src.size().width - 1) {
				src.ptr<double>(x + 1, y + 1)[k] += error[k] * 1 / 17.0;
			}
		}
	}
}

Mat dithering_parallel(Mat src) {
	Mat dst(src.size(), CV_8UC3);
	src.convertTo(src, CV_64FC3);
	int times = (src.rows - 1) * 2 + src.cols;
	int* coord_x = new int[src.rows * 2 - 1];
	int* coord_y = new int[src.rows * 2 - 1];
	omp_set_num_threads(2);
	#pragma omp parallel for 
	for (int i = 0; i < src.rows * 2 - 1; i += 2) {
		coord_x[i] = i / 2;
		coord_y[i] = 0;
	}
	for (int t = 0; t < times; t++) {
		int k = (t >= src.rows * 2) ? src.rows * 2 - 2 : t;
		int s = (t >= src.cols) ? ((t - src.cols) / 2 + 1) * 2 : 0;
		s = (s >= src.rows * 2) ? src.rows * 2 - 2 : s;
		if (s % 2 != 0) {
			std::cout << s;
			system("pause");
		}
		omp_set_num_threads(2);
		#pragma omp parallel for 
		for (int i = s; i <= k; i += 2) {
			processPiexl(src, dst, coord_x[i], coord_y[i]++);
		}
	}
	return dst;
}
int main() {
	LARGE_INTEGER Freq;
	LARGE_INTEGER Start;
	LARGE_INTEGER End;
	long used;

	Mat image;
	std::string path = "./InputImage/";
	std::string opath = "./OutputImage/";
	std::string inputFileName = "pandas";
	image = imread(path + inputFileName + ".jpg", 1);

	if (image.empty()) {
		std::cout << "load image error" << std::endl;
		system("pause");
		return -1;
	}
	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("original", image);
	
	KmeansImage_parallel(image);
	QueryPerformanceFrequency(&Freq);
	QueryPerformanceCounter(&Start);
	//-------------------------------
	Mat res = dithering(image);
	//-------------------------------
	QueryPerformanceCounter(&End);
	used = (((End.QuadPart - Start.QuadPart) * 1000) / Freq.QuadPart);
	std::cout << "Dithering" << std::endl;
	std::cout << "Cost time: " << used << " millisecond " << std::endl;
	namedWindow("Dithering", WINDOW_AUTOSIZE);
	imshow("Dithering", res);
	imwrite(opath + inputFileName + "_D.tif", res);
	
	
	QueryPerformanceFrequency(&Freq);
	QueryPerformanceCounter(&Start);
	//-------------------------------
	Mat res1 = dithering_parallel(image);
	//-------------------------------
	QueryPerformanceCounter(&End);
	used = (((End.QuadPart - Start.QuadPart) * 1000) / Freq.QuadPart);
	std::cout << "Parallel Dithering" << std::endl;
	std::cout << "Cost time: " << used << " millisecond " << std::endl;
	namedWindow("Dithering_Parallel", WINDOW_AUTOSIZE);
	imshow("Dithering_Parallel", res1);
	imwrite(opath + inputFileName + "_D_parallel.tif", res1);

	waitKey(0);
	system("pause");
	return 0;
}