#pragma once
#include<opencv2/opencv.hpp>
#include"warp_corner.h"


class Warper {
private:
	int mid = 0; 
public:
	vector<Mat> imgs, Hs; 
	float global_min_x = 0, global_min_y = 0; 
	float global_max_x = 0, global_max_y = 0;

	vector<WarpCorner> warpCorners; 
	vector<WarpImageInfo> warp_image_infos; 


	void myWarper();
	WarpImageInfo warp_info_by_H(int img_p, Mat H);
	void globalMinMaxXY();

	void weight_average(WarpImageInfo w1, WarpImageInfo w2, Mat& fusion);
	Mat cylindrical_projection(Mat srcImg);

	void get_imgs_hs_test();
	void cut_common_area_test();
};