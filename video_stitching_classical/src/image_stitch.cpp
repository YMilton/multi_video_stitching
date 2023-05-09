#include"image_stitch.h"

#include"feature_finder.h"
#include"optimal_seam.h"

#include"warper.h"
#include"camera.h"

#include"my_util.h"

Point2f ImageStitch::warpPoint(Point2f p, Mat H)
{
	Mat_<float> H_ = H;
	float x_ = (H_(0, 0) * p.x + H_(0, 1) * p.y + H_(0, 2)) / (H_(2, 0) * p.x + H_(2, 1) * p.y + H_(2, 2));
	float y_ = (H_(1, 0) * p.x + H_(1, 1) * p.y + H_(1, 2)) / (H_(2, 0) * p.x + H_(2, 1) * p.y + H_(2, 2));
	return Point2f(x_, y_);
}

Point2f* ImageStitch::warpSize(Mat src, Mat H)
{
	Point2f* ps = new Point2f[2];
	vector<Point> corners = { Point(0,0), Point(src.cols,0),Point(0,src.rows),Point(src.cols,src.rows) };
	Point2f pmax = warpPoint(corners[0], H), pmin = pmax;
	for (size_t i = 1; i < corners.size(); i++) {
		Point2f p_ = warpPoint(corners[i], H);
		pmax.x = p_.x > pmax.x ? p_.x : pmax.x;
		pmax.y = p_.y > pmax.y ? p_.y : pmax.y;
		pmin.x = p_.x < pmin.x ? p_.x : pmin.x;
		pmin.y = p_.y < pmin.y ? p_.y : pmin.y;
	}
	ps[0] = pmin;
	ps[1] = pmax;
	return ps;
}

Mat ImageStitch::direct_stitch(Mat src, Mat dst)
{
	if (cylindrical) {
		Warper w;
		src = w.cylindrical_projection(src);
		dst = w.cylindrical_projection(dst);
	}

	FeatureFinder finder;
	double t = getTickCount();
	Mat H_ = finder.find_H(src, dst, 3);
	cout << H_ << endl;
	cout << "scale: " << (getTickCount() - t) / getTickFrequency() << endl;

	t = getTickCount();
	Mat H = finder.find_H(src, dst);
	cout << H << endl;
	cout << "no scale: " << (getTickCount() - t) / getTickFrequency() << endl;

	Point2f* vec = warpSize(src, H);
	Point2f pmin = vec[0], pmax = vec[1];

	Mat warp;
	warpPerspective(src, warp, H, Size(pmax.x, pmax.y));
	
	Mat pano = average_weight(dst, warp, pmin);

	return pano;
}

Mat ImageStitch::average_weight(Mat dst, Mat warp, Point pmin)
{
	Mat pano = Mat::zeros(max(dst.rows, warp.rows), max(dst.cols, warp.cols), CV_8UC3);
	warp.copyTo(pano(Rect(0, 0, warp.cols, warp.rows)));
	dst.copyTo(pano(Rect(0, 0, dst.cols, dst.rows)));

	int threshold = 10;
	double alpha = 1;
	for (int i = 0; i < min(dst.rows, warp.rows); i++) {
		uchar* t = dst.ptr<uchar>(i);  
		uchar* w = warp.ptr<uchar>(i);
		uchar* p = pano.ptr<uchar>(i);
		for (int j = pmin.x; j < dst.cols; j++) {
			if (w[j * 3] <= threshold && w[j * 3 + 1] <= threshold && w[j * 3 + 2] <= threshold) {
				alpha = 1;
			}
			else {
				alpha = (float(dst.cols) - j) / (float(dst.cols) - pmin.x);
			}
			p[j * 3] = t[j * 3] * alpha + w[j * 3] * (1 - alpha);
			p[j * 3 + 1] = t[j * 3 + 1] * alpha + w[j * 3 + 1] * (1 - alpha);
			p[j * 3 + 2] = t[j * 3 + 2] * alpha + w[j * 3 + 2] * (1 - alpha);
		}
	}
	return pano;
}

Mat ImageStitch::seam_stitch(Mat src, Mat dst)
{
	if (cylindrical) {
		Warper w;
		src = w.cylindrical_projection(src);
		dst = w.cylindrical_projection(dst);
	}

	FeatureFinder finder;
	Mat H = finder.find_H(src, dst, 3);

	Point2f* vec = warpSize(src, H);
	Point2f pmin = vec[0], pmax = vec[1];

	Mat warp;
	warpPerspective(src, warp, H, Size(pmax.x, pmax.y));
	Mat pano = fusion_by_seam(dst, warp, pmin);

	return pano;
}

Mat ImageStitch::fusion_by_seam(Mat dst, Mat warp, Point pmin)
{
	Mat target_ = Mat::zeros(max(warp.rows, dst.rows), dst.cols, CV_8UC3);
	dst.copyTo(target_(Rect(0, 0, dst.cols, dst.rows)));
	Mat warp_ = Mat::zeros(max(warp.rows, dst.rows), warp.cols, CV_8UC3);
	warp.copyTo(warp_(Rect(0, 0, warp.cols, warp.rows)));
	Mat cut1 = target_(Rect(pmin.x, 0, target_.cols - pmin.x, target_.rows));
	Mat cut2 = warp_(Rect(pmin.x, 0, target_.cols - pmin.x, warp_.rows));

	OptimalSeam os;
	os.DP_find_seam(cut1, cut2);
	Mat fusion;
	os.DP_fusion_by_seam(cut1, cut2, fusion);
	target_.copyTo(warp_(Rect(0, 0, target_.cols, target_.rows)));
	fusion.copyTo(warp_(Rect(pmin.x, 0, fusion.cols, fusion.rows)));

	return warp_;
}

void ImageStitch::image_stitch_test()
{
	Mat img1 = imread("../multi_video_stitching/images/temple1.jpg");
	Mat img2 = imread("../multi_video_stitching/images/temple2.jpg");

	cylindrical = true;
	Mat pano1 = direct_stitch(img2, img1);
	MyUtil::showImg("direct_stitch", pano1);
	Mat pano2 = seam_stitch(img2, img1);
	MyUtil::showImg("seam_stitch", pano2);
	imwrite("pano1.jpg", pano1);
	imwrite("pano2.jpg", pano2);
	waitKey();
}





void ImageStitch::mid_stitch(vector<Mat> imgs, vector<Mat> Hs, Mat& pano)
{
	Warper warper;
	double t = getTickCount();
	for (int i = 0; i < imgs.size(); i++) {
		imgs[i] = warper.cylindrical_projection(imgs[i]);
	}
	cout << "2.1 cylindrical projection time: " << (getTickCount() - t) / getTickFrequency() << endl;
	warper.imgs = imgs;
	warper.Hs = Hs;
	t = getTickCount();
	warper.myWarper();
	cout << "2.2 warp image time: " << (getTickCount() - t) / getTickFrequency() << endl;

	Mat stitched(ceil(warper.global_max_y - warper.global_min_y), ceil(warper.global_max_x - warper.global_min_x), CV_8UC3);
	Mat warpImg = warper.warp_image_infos[0].warpImg;
	WarpCorner corner = warper.warp_image_infos[0].warpCorner;
	warpImg.copyTo(stitched(Rect(corner.min_x, 0, warpImg.cols, warpImg.rows)));

	t = getTickCount();
	for (int i = 1; i < warper.warp_image_infos.size(); i++) {
		warpImg = warper.warp_image_infos[i].warpImg;
		corner = warper.warp_image_infos[i].warpCorner;

		Mat fusion;
		WarpImageInfo w1, w2;
		w1 = warper.warp_image_infos[i - 1];
		w2 = warper.warp_image_infos[i];

		t = getTickCount();
		OptimalSeam os;
		if (tmp_seam.size() != Hs.size()) {
			os.seam_cut_fusion(w1, w2, fusion);
			tmp_seam.push_back(os.seam_paths);
			seam_Egs.push_back(os.Eg);
		}
		else {
			float x_left = w2.warpCorner.min_x;
			float x_right = w1.warpCorner.max_x;
			int common_width = x_right - x_left;
			Mat cut1 = w1.warpImg(Range::all(), Range(w1.warpImg.cols - common_width, w1.warpImg.cols)); 
			Mat cut2 = w2.warpImg(Range::all(), Range(0, common_width)); 

			vector<float> original = os.seam_gradient_value(tmp_seam[i - 1], seam_Egs[i - 1]);
			os.DP_find_seam(cut1, cut2);
			vector<float> current = os.seam_gradient_value(os.seam_paths, os.Eg);
			bool is_change = os.change_detect(original, current);

			if (is_change) { 
				os.DP_fusion_by_seam(cut1, cut2, fusion);
				tmp_seam[i - 1] = os.seam_paths;
				seam_Egs[i - 1] = os.Eg;
			}
			else {
				os.seam_paths = tmp_seam[i - 1];
				os.DP_fusion_by_seam(cut1, cut2, fusion);
			}
		}
		cout << "fusion with optimal seam time: " << (getTickCount() - t) / getTickFrequency() << endl; 

		fusion.copyTo(warpImg(Rect(0, 0, fusion.cols, fusion.rows)));
		warper.warp_image_infos[i].warpImg = warpImg;
		warpImg.copyTo(stitched(Rect(corner.min_x, 0, warpImg.cols, warpImg.rows)));
	}
	cout << "2.3 all find seam and fusion time: " << (getTickCount() - t) / getTickFrequency() << endl;
	pano = stitched;
}


void ImageStitch::video_stitch(vector<string> paths)
{
	vector<VideoCapture*> caps;
	double t = getTickCount();
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
	cout << "create videoCapture time: " << (getTickCount() - t) / getTickFrequency() << endl;

	t = getTickCount();
	Camera c; 
	vector<Mat> frames;
	if (c.grab_cameras(caps)) {
		frames = c.getFrames(caps);
	}
	vector<Mat> Hs;
	Warper warper;
	FeatureFinder finder;
	for (int i = 1; i < caps.size(); i++) {
		Mat frame = warper.cylindrical_projection(frames[i]);
		Mat target = warper.cylindrical_projection(frames[i - 1]);
		Mat H = finder.find_H(frame, target);
		cout << "H" << i - 1 << "-" << i << ":" << H << endl;
		Hs.push_back(H);
	}
	cout << "1.find homography time: " << (getTickCount() - t) / getTickFrequency() << endl << endl;

	VideoWriter vw;
	int i = 0;
	for (;;)
	{
		vector<Mat> frames;
		if (c.grab_cameras(caps)) {
			frames = c.getFrames(caps);
		}

		Mat pano;
		t = getTickCount();
		mid_stitch(frames, Hs, pano);
		cout << "2.image stitch time: " << (getTickCount() - t) / getTickFrequency() << endl << endl;
		if (i == 0) { 
			vw.open("pano.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0,
				Size(pano.cols, pano.rows), true);
			vw << pano;
		}
		else {

			vw << pano;
		}

		MyUtil::showImg("pano", pano);

		if (waitKey(33) == 27) break;
		i++;
	}

	for (int i = 0; i < caps.size(); i++) {
		delete caps[i];
	}
}


void ImageStitch::test_videos_stitch()
{
	vector<string> paths;



	paths.push_back("../multi_video_stitching/clip/camera0.avi");
	paths.push_back("../multi_video_stitching/clip/camera1.avi");
	paths.push_back("../multi_video_stitching/clip/camera2.avi");
	paths.push_back("../multi_video_stitching/clip/camera3.avi");



	video_stitch(paths);
}