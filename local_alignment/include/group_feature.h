#pragma once
#include<opencv.hpp>

using namespace cv;
using namespace std;

class SuperPixel {
public:
	Mat_<int> mask, lables; 
	Mat sp_img;  
	int num; 
};

class SuperPixelFeature {
public:
	Mat_<int> label_matrix;
	vector<int> labels;
	vector<vector<Point2f>> classify_s, classify_d;
};

class GroupFeature
{
private:

	static bool my_comp(const pair<int, int>& map1, const pair<int, int>& map2) { return map1.second > map2.second; }

	bool is_neighbor(vector<int> vec_labels, int label, Mat_<int> label_matrix);
	bool is_neighbor(int label1, int label2, Mat_<int> label_matrix);

public:
	vector<vector<Point2f>> groups_src, groups_dst; 

	void kmean_group(Mat src, Mat dst, int K, bool is_show);


	void superpixel_group(Mat src, Mat dst, bool is_show);
	SuperPixelFeature features_into_superpixel(Mat src, Mat dst, bool is_show);
	SuperPixel superpixel_slic(Mat img);

	void test();
};

