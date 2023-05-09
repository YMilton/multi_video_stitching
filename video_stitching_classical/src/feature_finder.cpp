#include<opencv2\xfeatures2d\nonfree.hpp>

#include"feature_finder.h"
#include"my_util.h"


FeatureFinder::FeatureFinder()
{
	this->detector = xfeatures2d::SIFT::create(1000);
	this->matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
}

FeatureFinder::FeatureFinder(string detector_name, string matcher_name)
{
	if (detector_name=="sift")
		this->detector = xfeatures2d::SIFT::create(1000); 
	else if (detector_name == "surf")
		this->detector = xfeatures2d::SurfFeatureDetector::create(1500);
	else if(detector_name == "orb")
		this->detector = ORB::create(800);
	else
		this->detector = NULL;

	if (matcher_name == "brute_force")
		this->matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	else if(matcher_name == "flann_based")
		this->matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	else
		this->matcher = NULL;
}

vector<vector<Point2f>> FeatureFinder::match_points(Mat src, Mat dst, bool is_show)
{
	if (src.dims == 0 || dst.dims == 0) {
		cout << "Error: the image's dims is zero." << endl;
		return vector<vector<Point2f>>();
	}

	Mat gray_s, gray_d;
	cvtColor(src, gray_s, COLOR_RGB2GRAY);
	cvtColor(dst, gray_d, COLOR_RGB2GRAY);

	vector<KeyPoint> keypoint_s, keypoint_d;
	Mat descriptor_s, descriptor_d;
	this->detector->detectAndCompute(gray_s, cv::noArray(), keypoint_s, descriptor_s);
	this->detector->detectAndCompute(gray_d, cv::noArray(), keypoint_d, descriptor_d);

	vector<vector<DMatch>> knn_matches;
	vector<DMatch> good_matches;
	matcher->knnMatch(descriptor_s, descriptor_d, knn_matches, 2);
	for (int i = 0; i < knn_matches.size(); i++) {
		if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	cout << "good matches size: " << good_matches.size() << endl;

	vector<vector<Point2f>> match_pair(2);
	for (size_t i = 0; i < good_matches.size(); i++) {
		match_pair[0].push_back(keypoint_s[good_matches[i].queryIdx].pt); 
		match_pair[1].push_back(keypoint_d[good_matches[i].trainIdx].pt); 
	}

	if (is_show) {
		Mat match = draw_matches(src, dst, match_pair);
		MyUtil::showImg("match", match);
	}

	return match_pair;
}


Mat FeatureFinder::find_H(Mat src, Mat dst)
{
	vector<vector<Point2f>> match_pair = match_points(src, dst, false);
	Mat H = findHomography(match_pair[0], match_pair[1], RANSAC);

	return H;
}

Mat FeatureFinder::find_H(Mat src, Mat dst, int scale)
{
	Mat src_, dst_;
	resize(src, src_, Size(src.cols / scale, src.rows / scale));
	resize(dst, dst_, Size(dst.cols / scale, dst.rows / scale));
	vector<vector<Point2f>> match_pair = match_points(src_, dst_, false);
	Mat H_ = findHomography(match_pair[0], match_pair[1], RANSAC);

	Mat_<float> S = Mat::eye(3, 3, CV_32F);
	S(0, 0) = 1.0 / scale;
	S(1, 1) = 1.0 / scale;

	H_.convertTo(H_, CV_32F);
	Mat H = S.inv() * H_ * S;

	return H;
}


Mat FeatureFinder::draw_matches(Mat src, Mat dst, vector<vector<Point2f>> match_points)
{
	Mat bg(src.rows, 2 * src.cols, CV_8UC3);
	src.copyTo(bg(Rect(src.cols, 0, src.cols, src.rows)));
	dst.copyTo(bg(Rect(0, 0, dst.cols, dst.rows)));

	for (int i = 0; i < match_points[0].size(); i++) {
		RNG rng(getTickCount());
		Scalar c = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

		circle(bg, match_points[1][i], 3, c, 2);
		circle(bg, match_points[0][i] + Point2f(src.cols, 0), 3, c, 2);
		line(bg, match_points[1][i], match_points[0][i] + Point2f(src.cols, 0), c);
	}

	return bg;
}


Mat FeatureFinder::draw_matches(Mat src, Mat dst, vector<vector<Point2f>> match_points, vector<uchar> mask)
{
	Mat bg(src.rows, 2 * src.cols, CV_8UC3);
	src.copyTo(bg(Rect(src.cols, 0, src.cols, src.rows)));
	dst.copyTo(bg(Rect(0, 0, dst.cols, dst.rows)));

	for (int i = 0; i < match_points[0].size(); i++) {
		if (mask[i]) {
			RNG rng(getTickCount());
			Scalar c = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

			circle(bg, match_points[1][i], 3, c, 2);
			circle(bg, match_points[0][i] + Point2f(src.cols, 0), 3, c, 2);
			line(bg, match_points[1][i], match_points[0][i] + Point2f(src.cols, 0), c);
		}
	}

	return bg;
}
