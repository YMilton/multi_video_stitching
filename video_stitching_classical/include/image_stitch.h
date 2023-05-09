#pragma once
#include<opencv.hpp>

using namespace cv;
using namespace std;

class ImageStitch
{
private:
	vector<vector<int>> tmp_seam; 
	vector<Mat> seam_Egs; 

public:
	bool cylindrical = false;

	Point2f warpPoint(Point2f p, Mat H);
	Point2f* warpSize(Mat src, Mat H);

	Mat direct_stitch(Mat src, Mat dst);
	Mat average_weight(Mat dst, Mat warp, Point pmin);

	Mat seam_stitch(Mat src, Mat dst);
	Mat fusion_by_seam(Mat dst, Mat warp, Point pmin);

	void image_stitch_test();


	void mid_stitch(vector<Mat> imgs, vector<Mat> Hs, Mat& pano);

	void video_stitch(vector<string> paths);
	void test_videos_stitch();
};

