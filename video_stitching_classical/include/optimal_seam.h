#pragma once
#include<opencv2\opencv.hpp>
#include"warp_corner.h"

using namespace cv;
using namespace std;

class OptimalSeam
{
private:
	void intensity_function(Mat I1, Mat I2, Mat& C);
	void gradient_function(Mat I1, Mat I2, Mat& G);
	void energy_function(Mat I1, Mat I2, Mat& E);

	Mat_<uchar> overlap, labels; 
	void computeCosts(Mat I1, Mat I2, Mat_<float>& E); 
	void computeStartEnd(Mat I1, Mat I2, Point &start, Point &end); 
	bool neighborhood_class(Point point, int class_num=2);

	int minCostIndex(float left, float mid, float right, int correct_index);

	float square(float x) { return x * x; }
	float distance(Point p1, Point p2) { return sqrtf(square(p1.x - p2.x) + square(p1.y - p2.y)); }
	float diffL2Square3(Mat img1, int x1, int y1, Mat img2, int x2, int y2);

public:
	Mat Ec, Eg;
	pair<vector<int>, vector<int>> seam_paths_pair; 
	vector<int> seam_paths, seam_paths_shrink; 

	void seam_cut_fusion(WarpImageInfo w1, WarpImageInfo w2, Mat& fusion);

	void seam_curve(Mat common_targ, Mat common_src);

	void fusion_by_seam(Mat I1, Mat I2, Mat& fusion);

	vector<float> seam_gradient_value(vector<int> seam_paths, Mat Eg);

	bool change_detect(vector<float> original, vector<float> current);

	void test();

	void DP_find_seam(Mat I1, Mat I2);
	void DP_fusion_by_seam(Mat I1, Mat I2, Mat& fusion);
	void showSeam(Mat I1, Mat I2);
	void showAllSeams(Mat I1, Mat I2, Mat_<uchar> control);


	void find_seam(Mat I1, Mat I2);
};

