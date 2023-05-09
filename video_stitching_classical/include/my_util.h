#pragma once
#include<opencv.hpp>

using namespace cv;
using namespace std;


class MyUtil
{
public:
	static void showImg(string winName, Mat img);
	static void createDir(string path);
	static void write_mat(string fileName, Mat mat);

	static float euclid(Point2f p1, Point2f p2);

	static float L2_square(Point2f p1, Point2f p2);
};

