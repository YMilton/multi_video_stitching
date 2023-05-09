#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;
using namespace cv::detail;

namespace stitch {
	class ImageStitch
	{
	private:
		Ptr<Feature2D> finder; 
		vector<CameraParams> cameras; 
		vector<Mat> warpImgs, warpMasks; 
		vector<Point> topCorners; 
	public:
		enum FEATURE{SIFT,SURF,ORB};
		vector<Mat> imgs; 
		Mat pano; 

		ImageStitch();
		ImageStitch(int detector);

		void getCameraArgs();

		void warpImages();

		void findSeam();

		void blendImages();

		Mat stitch();
	};
}


