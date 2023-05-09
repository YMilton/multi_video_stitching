#pragma once
#include<opencv.hpp>

using namespace cv;
using namespace std;

class WarpCorner
{
private:
	Point2f left_top, left_bottom;
	Point2f right_top, right_bottom;

public:

	float min_x = 0, max_x = 0, min_y = 0, max_y = 0; 

	WarpCorner() {};
	WarpCorner(Point2f left_top, Point2f left_bottom, Point2f right_top, Point2f right_bottom);

	void perspectiveCorner(Mat H);
	void perspectiveCornerInitial(Mat img, Mat H);

	void printCorners();
	void printMinMaxXY();
};


class WarpImageInfo
{
public:
	WarpCorner warpCorner;
	Mat warpImg;

	float min_x, max_x;
};


