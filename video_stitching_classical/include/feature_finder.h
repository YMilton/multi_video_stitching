
#pragma once
#include<opencv.hpp>

using namespace cv;
using namespace std;


class FeatureFinder
{
private:
	Ptr<Feature2D> detector; 
	Ptr<DescriptorMatcher> matcher; 

public:

	FeatureFinder();
	FeatureFinder(string detector_name, string matcher_name);

	vector<vector<Point2f>> match_points(Mat src, Mat dst, bool is_show);
	Mat find_H(Mat src, Mat dst);

	Mat find_H(Mat src, Mat dst, int scale);

	Mat draw_matches(Mat src, Mat dst, vector<vector<Point2f>> match_points);

	Mat draw_matches(Mat src, Mat dst, vector<vector<Point2f>> match_points, vector<uchar> mask);
};