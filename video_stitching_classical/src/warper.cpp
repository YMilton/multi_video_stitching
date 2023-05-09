#include"warper.h"
#include"my_util.h"
#include"optimal_seam.h"


void Warper::myWarper()
{
	globalMinMaxXY(); 
	warp_image_infos = vector<WarpImageInfo>(imgs.size());

	Mat H_ = Mat::eye(Size(3, 3), CV_32FC1);
	Mat H = H_.clone();

	for (int i = mid + 1; i < imgs.size(); i++) {
		Hs[i - 1].convertTo(Hs[i - 1], CV_32FC1);
		H = H * Hs[i - 1];
		warp_image_infos[i] = warp_info_by_H(i, H); 
	}
	warp_image_infos[mid] = warp_info_by_H(mid, H_);

	H = H_.clone();
	for (int i = mid - 1; i >= 0; i--) {
		Hs[i].convertTo(Hs[i], CV_32FC1);
		H = H * Hs[i].inv();
		warp_image_infos[i] = warp_info_by_H(i, H);
	}
}


WarpImageInfo Warper::warp_info_by_H(int i, Mat H)
{
	WarpImageInfo warp_image_info;
	Mat globalH = Mat::eye(Size(3, 3), CV_32FC1);
	if (global_min_y < 0) {
		globalH.at<float>(1, 2) = -global_min_y;
	}
	if (global_min_x < 0) { 
		globalH.at<float>(0, 2) = -global_min_x;
	}
	warpCorners[i].perspectiveCorner(globalH); 

	int w = warpCorners[i].max_x - warpCorners[i].min_x;
	int h = global_max_y - global_min_y; 

	Mat TxTy = Mat::eye(Size(3, 3), CV_32FC1);
	if (warpCorners[i].min_x < 0) { 
		TxTy.at<float>(0, 2) = abs(warpCorners[i].min_x);
	}
	if (warpCorners[i].min_x > 0) { 
		TxTy.at<float>(0, 2) = -abs(warpCorners[i].min_x);  
	}
	
	Mat warpImg;
	warpPerspective(imgs[i], warpImg, globalH * TxTy * H, Size(w, h));
	warp_image_info.warpImg = warpImg;
	warp_image_info.warpCorner = warpCorners[i];

	return warp_image_info;
}


void Warper::globalMinMaxXY()
{
	for (int i = 0; i < imgs.size(); i++){
		WarpCorner warperCorner(Point2f(0, 0), Point2f(0, imgs[i].rows),
			Point2f(imgs[i].cols, 0), Point2f(imgs[i].cols, imgs[i].rows));
		warpCorners.push_back(warperCorner);
	}

	mid = imgs.size() / 2;
	Mat H = Mat::eye(Size(3, 3), CV_32FC1);
	warpCorners[mid].perspectiveCorner(H);
	for (int i = mid + 1; i < imgs.size(); i++) {
		Hs[i - 1].convertTo(Hs[i - 1], CV_32FC1);
		H = H * Hs[i - 1];
		warpCorners[i].perspectiveCorner(H);
	}
	H = Mat::eye(Size(3, 3), CV_32FC1);
	for (int i = mid - 1; i >= 0; i--) {
		Hs[i].convertTo(Hs[i], CV_32FC1);
		H = H * Hs[i].inv();
		warpCorners[i].perspectiveCorner(H);
	}

	for (int i = 0; i < warpCorners.size(); i++) {
		global_min_x = (global_min_x > warpCorners[i].min_x ? warpCorners[i].min_x : global_min_x);
		global_max_x = (global_max_x < warpCorners[i].max_x ? warpCorners[i].max_x : global_max_x);
		global_min_y = (global_min_y > warpCorners[i].min_y ? warpCorners[i].min_y : global_min_y);
		global_max_y = (global_max_y < warpCorners[i].max_y ? warpCorners[i].max_y : global_max_y);
	}
}


void Warper::weight_average(WarpImageInfo w1, WarpImageInfo w2, Mat& fusion)
{
	int threshold = 10;
	float alpha = 1;
	float x_left = w2.warpCorner.min_x;
	float x_right = w1.warpCorner.max_x;
	int common_width = x_right - x_left;
	Mat cut1 = w1.warpImg(Range::all(), Range(w1.warpImg.cols - common_width, w1.warpImg.cols));
	Mat cut2 = w2.warpImg(Range::all(), Range(0, common_width));
	fusion = cut2.clone();
	for (int i = 0; i < fusion.rows; i++) {
		uchar* p1 = cut1.ptr<uchar>(i);  
		uchar* p2 = cut2.ptr<uchar>(i);
		uchar* f = fusion.ptr<uchar>(i);
		for (int j = x_left; j < x_right; j++) {
			int xx = j - x_left; 
			if (p1[xx * 3] <= threshold && p1[xx * 3 + 1] <= threshold && p1[xx * 3 + 2] <= threshold) {
				alpha = 1;
			}
			if (p2[xx * 3] <= threshold && p2[xx * 3 + 1] <= threshold && p2[xx * 3 + 2] <= threshold) {
				alpha = 0;
			}
			else {
				alpha = (j - x_left) / (x_right - x_left);
			}
			f[xx * 3] = p1[xx * 3] * (1 - alpha) + p2[xx * 3] * alpha;
			f[xx * 3 + 1] = p1[xx * 3 + 1] * (1 - alpha) + p2[xx * 3 + 1] * alpha;
			f[xx * 3 + 2] = p1[xx * 3 + 2] * (1 - alpha) + p2[xx * 3 + 2] * alpha;
		}
	}
}


Mat Warper::cylindrical_projection(Mat srcImg)
{
	int cols_hat, rows_hat;
	double f = srcImg.cols;
	double alpha = 2 * atan(srcImg.cols / (2 * f)); 
	cols_hat = f * alpha;
	rows_hat = srcImg.rows;

	Mat imgOut = Mat::zeros(rows_hat, cols_hat, CV_8UC3);


	double center_x = srcImg.cols / 2;
	double center_y = srcImg.rows / 2;
	for (int y = 0; y < imgOut.rows; y++) {
		for (int x = 0; x < imgOut.cols; x++) {
			int xx = tan(x / f - 0.5 * alpha) * f + center_x;
			int yy = (y - center_y) * sqrt((xx - center_x) * (xx - center_x) + f * f) / f + center_y;
			if (xx<0 || yy<0 || xx>srcImg.cols-1 || yy>srcImg.rows-1)continue;

			uchar* s = srcImg.ptr<uchar>(yy);
			uchar* o = imgOut.ptr<uchar>(y);
			for (int k = 0; k < 3; k++) {
				o[x * 3 + k] = s[xx * 3 + k];
			}
		}
	}
	return imgOut;
}


void Warper::get_imgs_hs_test()
{
	int img_size = 5;
	for (int i = 0; i < img_size; i++) {
		Mat img = imread("./images/boat" + to_string(i + 1) + ".jpg");
		imgs.push_back(img);
	}
	FileStorage fs("Hs_args.xml", FileStorage::READ);
	for (int i = 0; i < img_size - 1; i++) {
		Mat H;
		fs["H" + to_string(i + 1) + to_string(i + 2)] >> H;
		Hs.push_back(H);
	}
}


void Warper::cut_common_area_test()
{
	double t3 = (double)getTickCount();
	get_imgs_hs_test(); 
	myWarper();
	Mat stitched(ceil(global_max_y - global_min_y), ceil(global_max_x - global_min_x), CV_8UC3);

	Mat warpImg = warp_image_infos[0].warpImg;
	WarpCorner corner = warp_image_infos[0].warpCorner;
	
	warpImg.copyTo(stitched(Rect(corner.min_x, 0, warpImg.cols, warpImg.rows)));

	for (int i = 1; i < warp_image_infos.size(); i++) {
		warpImg = warp_image_infos[i].warpImg;
		corner = warp_image_infos[i].warpCorner;
		

		Mat fusion;
		WarpImageInfo w1, w2;
		w1 = warp_image_infos[i - 1];
		w2 = warp_image_infos[i];

		OptimalSeam os;
		os.seam_cut_fusion(w1, w2, fusion);

		fusion.copyTo(warpImg(Rect(0, 0, fusion.cols, fusion.rows)));
		warpImg.copyTo(stitched(Rect(corner.min_x, 0, warpImg.cols, warpImg.rows)));
		warp_image_infos[i].warpImg = warpImg;


	}

	MyUtil::showImg("stitched", stitched);

	double t4 = (double)getTickCount();
	cout << "elapsed time: " << (t4 - t3) / getTickFrequency() << "s" << endl;

	waitKey(0);
}
