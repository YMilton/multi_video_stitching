#pragma once
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace stitch {
	class VideoStitch
	{
	private:
		bool grabFrames(vector<VideoCapture*> caps);

		vector<Mat> getFrames(vector<VideoCapture*> caps);
	public:
		void imageStitch();

		void videoStitch();

		
	};

}

