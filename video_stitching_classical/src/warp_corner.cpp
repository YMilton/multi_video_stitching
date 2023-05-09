#include"warp_corner.h"


WarpCorner::WarpCorner(Point2f left_top, Point2f left_bottom, Point2f right_top, Point2f right_bottom)
{
	this->left_top = left_top;
	this->left_bottom = left_bottom;
	this->right_top = right_top;
	this->right_bottom = right_bottom;
}


void WarpCorner::perspectiveCorner(Mat H)
{
	float C[3][4] = { left_top.x,left_bottom.x,right_top.x, right_bottom.x,
					   left_top.x,left_bottom.y,right_top.y,right_bottom.y,
					   1,1,1,1 };
	Mat corners = Mat(3, 4, CV_32FC1, C);
	H.convertTo(H, CV_32FC1);

	Mat W = H * corners;
	this->left_top = Point2f(W.at<float>(0, 0) / W.at<float>(2, 0), W.at<float>(1, 0) / W.at<float>(2, 0));
	this->left_bottom = Point2f(W.at<float>(0, 1) / W.at<float>(2, 1), W.at<float>(1, 1) / W.at<float>(2, 1));
	this->right_top = Point2f(W.at<float>(0, 2) / W.at<float>(2, 2), W.at<float>(1, 2) / W.at<float>(2, 2));
	this->right_bottom = Point2f(W.at<float>(0, 3) / W.at<float>(2, 3), W.at<float>(1, 3) / W.at<float>(2, 3));

	min_x = left_top.x > left_bottom.x ? left_bottom.x : left_top.x;
	min_y = left_top.y > right_top.y ? right_top.y : left_top.y;
	max_x = right_top.x > right_bottom.x ? right_top.x : right_bottom.x;
	max_y = left_bottom.y > right_bottom.y ? left_bottom.y : right_bottom.y;
}

void WarpCorner::perspectiveCornerInitial(Mat img, Mat H)
{
	float C[3][4] = { 0,0,img.cols, img.cols,
					   0,img.rows,0,img.rows,
					   1,1,1,1 };
	Mat corners = Mat(3, 4, CV_32FC1, C);
	H.convertTo(H, CV_32FC1);

	Mat W = H * corners;
	this->left_top = Point2f(W.at<float>(0, 0) / W.at<float>(2, 0), W.at<float>(1, 0) / W.at<float>(2, 0));
	this->left_bottom = Point2f(W.at<float>(0, 1) / W.at<float>(2, 1), W.at<float>(1, 1) / W.at<float>(2, 1));
	this->right_top = Point2f(W.at<float>(0, 2) / W.at<float>(2, 2), W.at<float>(1, 2) / W.at<float>(2, 2));
	this->right_bottom = Point2f(W.at<float>(0, 3) / W.at<float>(2, 3), W.at<float>(1, 3) / W.at<float>(2, 3));

	min_x = left_top.x > left_bottom.x ? left_bottom.x : left_top.x;
	min_y = left_top.y > right_top.y ? right_top.y : left_top.y;
	
	max_x = right_top.x > right_bottom.x ? right_top.x : right_bottom.x;
	max_y = left_bottom.y > right_bottom.y ? left_bottom.y : right_bottom.y;
}

void WarpCorner::printCorners()
{
	cout << "left_top: " << left_top << "\t" << "left_bottom: " << left_bottom << "\t"
		<< "right_top: " << right_top << "\t" << "right_bottom: " << right_bottom << endl;
}

void WarpCorner::printMinMaxXY()
{
	cout << "(min_x,max_x): " << min_x << "," << max_x << "\t"
		<< "(min_y, max_y): " << min_y << "," << max_y << endl;
}
