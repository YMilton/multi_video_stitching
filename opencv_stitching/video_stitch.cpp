#include"video_stitch.h"
#include"stitch_detail.h"

namespace stitch {

	void VideoStitch::imageStitch() {
		int num = 6;
		vector<Mat> imgs;
		for (int i = 0; i < num; i++) {
			Mat img = imread("../multi_video_stitching/images/boat" + to_string(i + 1) + ".jpg");
			resize(img, img, img.size() / 5);
			imgs.push_back(img);
		}

		ImageStitch is;
		is.imgs = imgs;

		Mat pano = is.stitch();

		namedWindow("pano", WINDOW_NORMAL);
		imshow("pano", pano);
		waitKey(0);
	}

	void VideoStitch::videoStitch()
	{
		vector<string> paths;
		paths.push_back("3");
		paths.push_back("1");
		paths.push_back("0");
		


		vector<VideoCapture*> caps;
		for (int i = 0; i < paths.size(); i++) 
		{
			cv::VideoCapture* cap = new cv::VideoCapture();
			if (paths[i].length() == 1) {
				cap->open(stoi(paths[i]));
				cap->set(CAP_PROP_FRAME_WIDTH, 1280);
				cap->set(CAP_PROP_FRAME_HEIGHT, 960);
				caps.push_back(cap);
				cout << "open the camera" + to_string(i) + " success!" << endl;
			}
			else {
				cap->open(paths[i]);
				caps.push_back(cap);
			}
		}

		vector<Mat> frames;
		ImageStitch is;
		bool is_grab = grabFrames(caps);
		if (is_grab) {
			frames = getFrames(caps);
			is.imgs = frames;
			is.getCameraArgs(); 
		}

		VideoWriter vw;
		int i = 0;
		for (;;)
		{
			
			double t = getTickCount();
			vector<Mat> frames;
			bool is_grab = grabFrames(caps);
			if (is_grab) {
				frames = getFrames(caps);
			}
			else {
				cout << " grab frames fail." << endl;
				break;
			}
			cout << "get image frames time: " << (getTickCount() - t) / getTickFrequency() << endl;

			is.imgs = frames;
			is.warpImages();
			is.findSeam();
			is.blendImages();
			Mat pano = is.pano.clone();

			cout << "get pano time: " << (getTickCount() - t) / getTickFrequency() << endl << endl;

			namedWindow("pano", WINDOW_NORMAL);
			imshow("pano", pano);

			if (i == 0) {
				vw.open("pano.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 5.0,
					Size(pano.cols, pano.rows), true);
				vw << pano;
			}
			else {
				vw << pano;
			}

			if (waitKey(33) == 27) break;
			i++;
		}

		for (int i = 0; i < caps.size(); i++) {
			delete caps[i];
		}
	}

	bool VideoStitch::grabFrames(vector<VideoCapture*> caps)
	{
		bool isGrab = true;
		for (int i = 0; i < caps.size(); i++) {
			isGrab = isGrab && caps[i]->grab();
		}
		return isGrab;
	}

	vector<Mat> VideoStitch::getFrames(vector<VideoCapture*> caps)
	{
		vector<Mat> frames;
		for (int i = 0; i < caps.size(); i++) {
			Mat frame;
			caps[i]->retrieve(frame);
			frames.push_back(frame.clone());
		}
		return frames;
	}

}
