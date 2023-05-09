#include"optimal_seam.h"
#include"my_util.h"

void OptimalSeam::intensity_function(Mat I1, Mat I2, Mat& C)
{
	overlap = (I1 != 0) & (I2 != 0);
	C = 255.0 * 255.0 * 3.0 * Mat::ones(I1.size(), CV_64FC1);
	for (int i = 0; i < I1.rows; i++) {
		double* ptr_I1 = I1.ptr<double>(i);
		double* ptr_I2 = I2.ptr<double>(i);
		double* ptr_C = C.ptr<double>(i);
		for (int j = 0; j < I1.cols; j++) {
			if (j > 0 && overlap(i, j))
				ptr_C[j] = square(ptr_I1[j * 3] - ptr_I2[j * 3]) + square(ptr_I1[j * 3 + 1] - ptr_I2[j * 3 + 1])
				+ square(ptr_I1[j * 3 + 2] - ptr_I2[j * 3 + 2]);
		}
	}
}
void OptimalSeam::gradient_function(Mat I1, Mat I2, Mat& G)
{
	G = Mat::zeros(I1.size(), CV_64FC1);
	Mat g1, g2;
	I1.convertTo(I1, CV_8UC3);
	I2.convertTo(I2, CV_8UC3);
	cvtColor(I1, g1, COLOR_BGR2GRAY);
	cvtColor(I2, g2, COLOR_BGR2GRAY);
	for (int i = 0; i < G.rows; i++) {
		double* ptr_I1 = g1.ptr<double>(i);
		double* ptr_I2 = g2.ptr<double>(i);
		double* ptr_G = G.ptr<double>(i);
		for (int j = 1; j < G.cols; j++) {
			if (overlap(i, j)) {
				double gdx1 = abs(ptr_I1[j] - ptr_I1[j - 1]);
				double gdx2 = abs(ptr_I2[j] - ptr_I2[j - 1]);
				ptr_G[j] = square(gdx1) + square(gdx2);
			}
		}
	}
}

void OptimalSeam::energy_function(Mat I1, Mat I2, Mat& E)
{
	cvtColor(I1, I1, COLOR_BGR2GRAY);
	cvtColor(I2, I2, COLOR_BGR2GRAY);
	I1.convertTo(I1, CV_32FC1);
	I2.convertTo(I2, CV_32FC1);

	Mat Sx = (Mat_<float>(3, 3) << -2, 0, 2, -1, 0, 1, -2, 0, 2);
	Mat Sy = (Mat_<float>(3, 3) << -2, -1, -2, 0, 0, 0, 2, 1, 2);

	Mat delta_Sx, delta_Sy;
	filter2D(I1 - I2, delta_Sx, -1, Sx);
	filter2D(I1 - I2, delta_Sy, -1, Sy);
	Ec = (I1 - I2).mul(I1 - I2);
	Eg = (delta_Sx).mul(delta_Sx) + (delta_Sy).mul(delta_Sy);
	E = Ec + Eg;



}



void OptimalSeam::seam_cut_fusion(WarpImageInfo w1, WarpImageInfo w2, Mat& fusion)
{
	int x_left = w2.warpCorner.min_x;
	int x_right = w1.warpCorner.max_x;
	int common_width = x_right - x_left;
	Mat cut1 = w1.warpImg(Range::all(), Range(w1.warpImg.cols - common_width, w1.warpImg.cols)); 
	Mat cut2 = w2.warpImg(Range::all(), Range(0, common_width)); 

	DP_find_seam(cut1, cut2);

	DP_fusion_by_seam(cut1, cut2, fusion);
}

void OptimalSeam::seam_curve(Mat common_targ, Mat common_src)
{
	Mat EE;
	energy_function(common_targ, common_src, EE); 

	Mat E = EE.colRange(EE.cols / 6, EE.cols * 5 / 6);

	Mat paths_weight = E(Rect(0, 0, E.cols, 1));
	Mat paths(1, E.cols, CV_16UC1);
	ushort* p = paths.ptr<ushort>(0);
	for (int i = 0; i < E.cols; i++) p[i] = i;

	for (int i = 1; i < E.rows; i++)
	{
		Mat weights = Mat(1, E.cols, CV_64FC1);
		Mat path_row = Mat(1, E.cols, CV_16UC1);

		double* p = E.ptr<double>(i);  
		ushort* path_p = paths.ptr<ushort>(i - 1); 
		double* value_p = paths_weight.ptr<double>(0); 

		for (int j = 0; j < E.cols; j++)
		{
			int correct_index = path_p[j];
			double correct_upper_value = value_p[j];
			if (correct_index == 0) { 
				double mid = p[correct_index] + correct_upper_value;
				double right = p[correct_index + 1] + correct_upper_value;
				if (mid < right) {
					path_row.at<ushort>(0, j) = correct_index;
					weights.at<double>(0, j) = mid;
				}
				else {
					path_row.at<ushort>(0, j) = correct_index + 1;
					weights.at<double>(0, j) = right;
				}
			}
			else if (correct_index == E.cols - 1) { 
				double left = p[correct_index - 1] + correct_upper_value;
				double mid = p[correct_index] + correct_upper_value;
				if (mid < left) {
					path_row.at<ushort>(0, j) = correct_index;
					weights.at<double>(0, j) = mid;
				}
				else {
					path_row.at<ushort>(0, j) = correct_index - 1;
					weights.at<double>(0, j) = left;
				}
			}
			else { 
				double left = p[correct_index - 1] + correct_upper_value;
				double mid = p[correct_index] + correct_upper_value;
				double right = p[correct_index + 1] + correct_upper_value;
				if (left <= mid && left <= right) {
					path_row.at<ushort>(0, j) = correct_index - 1;
					weights.at<double>(0, j) = left;
				}
				else if (right <= mid && right <= left) {
					path_row.at<ushort>(0, j) = correct_index + 1;
					weights.at<double>(0, j) = right;
				}
				else {
					path_row.at<ushort>(0, j) = correct_index;
					weights.at<double>(0, j) = mid;
				}
			}
		}
		paths_weight = weights;
		paths.push_back(path_row);
	}

	Mat img = Mat::zeros(E.size(), CV_8U);
	for (int k = 0; k < paths.cols; k++) {
		vector<int> seam = (vector<int>)paths.colRange(k, k + 1);

		for (int y = 1; y < paths.rows; y++) {
			line(img, Point(seam[y], y), Point(seam[y - 1], y - 1), 255);
		}
	}

	MyUtil::showImg("seam", img);
	waitKey();

	int min_index = 0;
	double* pw = paths_weight.ptr<double>(0);
	for (int i = 0; i < paths_weight.cols - 1; i++) {
		min_index = pw[i] < pw[i + 1] ? i : i + 1;
	}

	seam_paths = (vector<int>)paths.colRange(min_index, min_index + 1);
}

void OptimalSeam::fusion_by_seam(Mat I1, Mat I2, Mat& fusion)
{
	int threshold = 10;
	float alpha = 1;
	float min_index = *min_element(seam_paths.begin(), seam_paths.end()); 
	float max_index = *max_element(seam_paths.begin(), seam_paths.end());
	fusion = I1.clone();

	for (int i = 0; i < fusion.rows; i++) {
		uchar* p1 = I1.ptr<uchar>(i);  
		uchar* p2 = I2.ptr<uchar>(i);
		uchar* f = fusion.ptr<uchar>(i);

		for (int j = 0; j < min_index; j++) {
			if (p1[j * 3] <= threshold && p1[j * 3 + 1] <= threshold && p1[j * 3 + 2] <= threshold) {
				f[j * 3] = p2[j * 3];
				f[j * 3 + 1] = p2[j * 3 + 1];
				f[j * 3 + 2] = p2[j * 3 + 2];
			}
			else {
				f[j * 3] = p1[j * 3];
				f[j * 3 + 1] = p1[j * 3 + 1];
				f[j * 3 + 2] = p1[j * 3 + 2];
			}
		}
		for (int j = max_index; j < I1.cols; j++) {
			if (p2[j * 3] <= threshold && p2[j * 3 + 1] <= threshold && p2[j * 3 + 2] <= threshold) {
				f[j * 3] = p1[j * 3];
				f[j * 3 + 1] = p1[j * 3 + 1];
				f[j * 3 + 2] = p1[j * 3 + 2];
			}
			else {
				f[j * 3] = p2[j * 3];
				f[j * 3 + 1] = p2[j * 3 + 1];
				f[j * 3 + 2] = p2[j * 3 + 2];
			}
		}

		for (int j = min_index; j < max_index; j++) {
			if (p1[j * 3] <= threshold && p1[j * 3 + 1] <= threshold && p1[j * 3 + 2] <= threshold) {
				alpha = 1;
			}
			else { 
				alpha = (j - min_index) / (max_index - min_index);
			}
			f[j * 3] = p1[j * 3] * (1 - alpha) + p2[j * 3] * alpha;
			f[j * 3 + 1] = p1[j * 3 + 1] * (1 - alpha) + p2[j * 3 + 1] * alpha;
			f[j * 3 + 2] = p1[j * 3 + 2] * (1 - alpha) + p2[j * 3 + 2] * alpha;
		}
	}

	for (int y = 0; y < seam_paths.size() - 1; y++) {
		line(fusion, Point(seam_paths[y], y), Point(seam_paths[y + 1], y + 1), Scalar(0, 0, 255), 1);
	}
}



vector<float> OptimalSeam::seam_gradient_value(vector<int> seam_paths, Mat Eg)
{
	vector<float> Gv;
	for (int i = 0; i < Eg.rows; i++) {
		float* p = Eg.ptr<float>(i);

		float value = p[seam_paths[i]];
		Gv.push_back(value);
	}
	return Gv;
}
bool OptimalSeam::change_detect(vector<float> original, vector<float> current)
{
	float delta = 0.5;
	int Ct = 0;
	for (int i = 0; i < original.size(); i++) {
		if (original[i] == 0) continue;
		float delta_gradient = (current[i] - original[i]) / original[i];
		if (delta_gradient > delta) Ct++;
	}
	if (Ct > 0.3 * original.size()) { 
		return true;
	}
	return false;
}


void OptimalSeam::test()
{
	Mat I1 = imread("./cut_img/cut1.jpg");
	Mat I2 = imread("./cut_img/cut2.jpg");
	Mat img1, img2;
	int size_times = 1;
	resize(I1, img1, Size(I1.cols / size_times, I1.rows / size_times));
	resize(I2, img2, Size(I2.cols / size_times, I2.rows / size_times));
	cout << img1.size() << endl;
	DP_find_seam(img1, img2);

	Mat fusion;
	DP_fusion_by_seam(img1, img2, fusion);
	MyUtil::showImg("fusion", fusion);
	waitKey();
}

void OptimalSeam::DP_find_seam(Mat I1, Mat I2)
{
	Mat_<float> E;
	computeCosts(I1, I2, E);
	E.convertTo(E, CV_32F);
	Mat_<uchar> control = Mat::zeros(E.size(), CV_8U); 
	Mat_<float> sum_cost = E.rowRange(0, 1); 

	for (int i = 1; i < E.rows; i++) {
		float* E_ptr = E.ptr<float>(i);
		Mat_<float> cost_tmp = sum_cost.clone();
		for (int j = 0; j < E.cols; j++) {
			float min_e = sum_cost(0, j);
			int c_x = 2; 
			if (j > 0 && min_e > sum_cost(0, j - 1)) {
				min_e = sum_cost(0, j - 1);
				c_x = 1;
			}
			if (j < E.cols - 1 && min_e > sum_cost(0, j + 1)) {
				min_e = sum_cost(0, j + 1);
				c_x = 3;
			}
			cost_tmp(0, j) = min_e + E_ptr[j];
			control(i, j) = c_x;
		}
		sum_cost = cost_tmp;
	}


	int end_x = control.cols / 2;
	Point current_p(end_x, control.rows - 1), top_p(end_x, control.rows - 1);
	seam_paths.push_back(end_x);
	for (; top_p.y != 0; seam_paths.push_back(top_p.x)) {
		if (control(current_p) == 1) top_p.x--;
		else if (control(current_p) == 3) top_p.x++;
		top_p.y--;
		current_p = top_p; 
	}
	reverse(seam_paths.begin(), seam_paths.end());


	
}

void OptimalSeam::DP_fusion_by_seam(Mat I1, Mat I2, Mat& fusion)
{
	int threshold = 10;
	float alpha = 1.0, min_idx, max_idx;
	min_idx = *min_element(seam_paths.begin(), seam_paths.end()); 
	max_idx = *max_element(seam_paths.begin(), seam_paths.end());

	int fusion_width = 10;
	fusion = Mat::zeros(I1.size(), CV_8UC3);


	for (int i = 0; i < I1.rows; i++) {
		int start = max(0, seam_paths[i] - fusion_width), end = min(I1.cols, seam_paths[i] + fusion_width);
		uchar* ptr1 = I1.ptr<uchar>(i);
		uchar* ptr2 = I2.ptr<uchar>(i);
		uchar* ptr_f = fusion.ptr<uchar>(i);

		for (int j = 0; j < start; j++) {
			if (ptr1[j * 3] <= threshold && ptr1[j * 3 + 1] <= threshold && ptr1[j * 3 + 2] <= threshold) {
				continue;
			}
			else {
				ptr_f[j * 3] = ptr1[j * 3];
				ptr_f[j * 3 + 1] = ptr1[j * 3 + 1];
				ptr_f[j * 3 + 2] = ptr1[j * 3 + 2];
			}
		}
		for (int j = end; j < I1.cols; j++) {
			if (ptr2[j * 3] <= threshold && ptr2[j * 3 + 1] <= threshold && ptr2[j * 3 + 2] <= threshold) {
				continue;
			}
			else {
				ptr_f[j * 3] = ptr2[j * 3];
				ptr_f[j * 3 + 1] = ptr2[j * 3 + 1];
				ptr_f[j * 3 + 2] = ptr2[j * 3 + 2];
			}
		}
		for (int j = start; j < end; j++) {
			if (ptr1[j * 3] < threshold && ptr1[j * 3 + 1] < threshold && ptr1[j * 3 + 2] < threshold) {
				alpha = 1.0;
			}
			else {
				alpha = float(j - start) / float(end - start); 
			}
			ptr_f[j * 3] = ptr1[j * 3] * (1 - alpha) + ptr2[j * 3] * alpha;
			ptr_f[j * 3 + 1] = ptr1[j * 3 + 1] * (1 - alpha) + ptr2[j * 3 + 1] * alpha;
			ptr_f[j * 3 + 2] = ptr1[j * 3 + 2] * (1 - alpha) + ptr2[j * 3 + 2] * alpha;
		}
	}
}

void OptimalSeam::showSeam(Mat I1, Mat I2)
{
	Mat g1, g2;
	cvtColor(I1, g1, COLOR_BGR2GRAY);
	cvtColor(I2, g2, COLOR_BGR2GRAY);
	labels = Mat::zeros(I1.size(), CV_8UC1);
	set<int> class_set;
	for (int i = 0; i < labels.rows; i++) {
		uchar* ptr_m1 = g1.ptr<uchar>(i);
		uchar* ptr_m2 = g2.ptr<uchar>(i);
		uchar* ptr_m = labels.ptr<uchar>(i);
		for (int j = 0; j < labels.cols; j++) {
			if (ptr_m1[j] != 0 && ptr_m2[j] != 0) {
				ptr_m[j] = 4;
			}
			else if (ptr_m1[j] != 0) {
				ptr_m[j] = 1;
			}
			else if (ptr_m2[j] != 0) {
				ptr_m[j] = 2;
			}
		}
	}
	Mat img = labels * 64;
	float min_idx = *min_element(seam_paths.begin(), seam_paths.end());
	float max_idx = *max_element(seam_paths.begin(), seam_paths.end());
	int fusion_width = 20;
	for (int i = 1; i < seam_paths.size(); i++) {
		line(img, Point(seam_paths[i], i), Point(seam_paths[i - 1], i - 1), Scalar(0,255,0));
	}

	MyUtil::showImg("seam", img);
}

void OptimalSeam::showAllSeams(Mat I1, Mat I2, Mat_<uchar> control)
{
	Mat g1, g2;
	cvtColor(I1, g1, COLOR_BGR2GRAY);
	cvtColor(I2, g2, COLOR_BGR2GRAY);
	labels = Mat::zeros(I1.size(), CV_8UC1);
	set<int> class_set;
	for (int i = 0; i < labels.rows; i++) {
		uchar* ptr_m1 = g1.ptr<uchar>(i);
		uchar* ptr_m2 = g2.ptr<uchar>(i);
		uchar* ptr_m = labels.ptr<uchar>(i);
		for (int j = 0; j < labels.cols; j++) {
			if (ptr_m1[j] != 0 && ptr_m2[j] != 0) {
				ptr_m[j] = 4;
			}
			else if (ptr_m1[j] != 0) {
				ptr_m[j] = 1;
			}
			else if (ptr_m2[j] != 0) {
				ptr_m[j] = 2;
			}
		}
	}
	Mat img = labels * 64;
	vector<int> seam_paths_;
	for (int end_x = 0; end_x < control.cols; end_x++) {
		Point current_p(end_x, control.rows - 1), top_p(end_x, control.rows - 1);
		seam_paths_.push_back(end_x);
		for (; top_p.y != 0; seam_paths_.push_back(top_p.x)) {
			if (control(current_p) == 1) top_p.x--;
			else if (control(current_p) == 3) top_p.x++;
			top_p.y--;
			line(img, current_p, top_p, 150);
			current_p = top_p; 
		}
	}
	MyUtil::showImg("allSeam", img);
}





void OptimalSeam::find_seam(Mat I1, Mat I2)
{
	Mat_<float> E;
	computeCosts(I1, I2, E);
	Point start, end;
	computeStartEnd(I1, I2, start, end);

	cout << start << endl;
	cout << end << endl << endl;;

	Mat_<uchar> reachable = Mat::zeros(I1.size(), CV_8UC1); 
	Mat_<uchar> control = Mat::zeros(I1.size(), CV_8UC1);
	Mat_<float> sum_cost = Mat::zeros(I1.size(), CV_32FC1);

	reachable(start) = 255; 
	sum_cost(start) = E(start);
	for (int y = start.y + 1; y < end.y; y++) {
		for (int x = 0; x < E.cols; x++) {
			vector<pair<float, int>> step;

			if (labels(y, x) == 4) { 
				if (reachable(y - 1, x))
					step.push_back(make_pair(sum_cost(y - 1, x) + E(y, x), 2));
				if (x > 0 && reachable(y - 1, x - 1))
					step.push_back(make_pair(sum_cost(y - 1, x - 1) + E(y, x), 1));
				if (x < E.cols - 1 && reachable(y - 1, x + 1))
					step.push_back(make_pair(sum_cost(y - 1, x + 1) + E(y, x), 3));
			}

			if (step.size() > 0) { 
				pair<float, int> opt = *min_element(step.begin(), step.end());
				sum_cost(y, x) = opt.first;
				control(y, x) = opt.second;
				reachable(y, x) = 255;
			}
		}
	}

	vector<Point> seam;
	seam.push_back(end);
	Point p = end;
	for (; p.y != start.y; seam.push_back(p)) {
		if (control(p) == 1) p.x--;
		else if (control(p) == 3) p.x++;
		p.y--;
	}

	MyUtil::write_mat("sum_cost.txt", sum_cost);
	MyUtil::write_mat("control.txt", control);
	imwrite("reachable.jpg", reachable);
	imwrite("control.jpg", control * 80);

	Mat img = labels * 60;
	for (int i = 1; i < seam.size(); i++) {
		line(img, seam[i], seam[i - 1], 0);
		cout << seam[i] << endl;
	}
	imwrite("seam.jpg", img);
}

void OptimalSeam::computeCosts(Mat I1, Mat I2, Mat_<float>& E)
{
	I1.convertTo(I1, CV_32FC3);
	I2.convertTo(I2, CV_32FC3);

	Mat gray1, gray2;
	cvtColor(I1, gray1, COLOR_BGR2GRAY);
	cvtColor(I2, gray2, COLOR_BGR2GRAY);

	Mat mask1 = gray1 != 0;
	Mat mask2 = gray2 != 0;
	overlap = mask1 & mask2;

	Mat_<float> gradx1, gradx2;
	Sobel(gray1, gradx1, CV_32F, 1, 0);
	Sobel(gray2, gradx2, CV_32F, 1, 0);

	Mat_<float> costs(overlap.size(), CV_32F);
	Mat_<float> costGrad(overlap.size(), CV_32F);
	for (int y = 0; y < overlap.rows; y++) {
		for (int x = 0; x < overlap.cols; x++) {
			if (x > 0) {
				costGrad(y, x) = abs(gradx1(y, x)) + abs(gradx1(y, x - 1)) + abs(gradx2(y, x)) + abs(gradx2(y, x - 1));
			}
			
			if (x>0 && overlap(y, x) && overlap(y, x - 1)) { 
				float costColor = (diffL2Square3(I1, x - 1, y, I2, x, y) +
					diffL2Square3(I1, x, y, I2, x - 1, y)) / 2;  

				costs(y, x) = costColor / costGrad(y,x);
			}
			else {
				costs(y, x) = 255 * 255 * 3; 
			}
		}
	}
	E = costs.clone();
	Eg = costGrad.clone();
}

void OptimalSeam::computeStartEnd(Mat I1, Mat I2, Point& start, Point& end)
{
	Mat gray1, gray2;
	cvtColor(I1, gray1, COLOR_BGR2GRAY);
	cvtColor(I2, gray2, COLOR_BGR2GRAY);

	labels = Mat::zeros(I1.size(), CV_8UC1);
	set<int> class_set;
	for (int i = 0; i < labels.rows; i++) {
		uchar* ptr_m1 = gray1.ptr<uchar>(i);
		uchar* ptr_m2 = gray2.ptr<uchar>(i);
		uchar* ptr_m = labels.ptr<uchar>(i);
		for (int j = 0; j < labels.cols; j++) {
			if (ptr_m1[j] != 0 && ptr_m2[j] != 0) {
				ptr_m[j] = 4;
				class_set.insert(4);
			}
			else if (ptr_m1[j] != 0) {
				ptr_m[j] = 1;
				class_set.insert(1);
			}
			else if (ptr_m2[j] != 0) {
				ptr_m[j] = 2;
				class_set.insert(2);
			}
		}
	}

	cout << "class_set: " << class_set.size() << endl;
	for (set<int>::iterator iter = class_set.begin(); iter != class_set.end(); ++iter) {
		cout << *iter << endl;
	}

	imwrite("labels.jpg", labels * 60);

	if (class_set.size() <= 2) {
		Point rt(I1.cols, 0), lb(0, I1.rows);
		float min_rt = numeric_limits<float>::max();
		float min_lb = min_rt;
		for (int y = 0; y < labels.rows; y++) {
			for (int x = 0; x < labels.cols; x++) {
				if (labels(y, x) == 4) {
					if (distance(Point(x, y), rt) < min_rt) {
						min_rt = distance(Point(x, y), rt);
						start = Point(x, y);
					}

					if (distance(Point(x, y), lb) < min_lb) {
						min_lb = distance(Point(x, y), lb);
						end = Point(x, y);
					}
				}
			}
		}
		return;
	}

	vector<Point> edges;
	for (int y = 0; y < labels.rows; y++) {
		for (int x = 0; x < labels.cols; x++) {
			if (labels(y, x) == 4) {
				if (x > 0 && labels(y, x - 1) != 4) edges.push_back(Point(x, y));

				if (x < labels.cols - 1 && labels(y, x + 1) != 4) edges.push_back(Point(x, y));

				if (y > 0 && labels(y - 1, x) != 4) edges.push_back(Point(x, y));

				if (y < labels.rows - 1 && labels(y + 1, x) != 4) edges.push_back(Point(x, y));
			}
		}
	}


	vector<Point> edge_nodes;
	for (int i = 0; i < edges.size(); i++) {
		if (neighborhood_class(edges[i])) edge_nodes.push_back(edges[i]);
	}


	vector<int> bestLabels;
	float l2distance = square(10); 
	cv::partition(edge_nodes, bestLabels, [l2distance, this](Point p1, Point p2) { 
		return distance(p1, p2) < l2distance;
		});
	int num_labels = *max_element(bestLabels.begin(), bestLabels.end()) + 1; 


	if (num_labels >= 2)
	{
		vector<Point> sum_point(num_labels);
		vector<vector<Point>> category_points(num_labels);
		for (int i = 0; i < edge_nodes.size(); i++) {
			sum_point[bestLabels[i]] += edge_nodes[i];
			category_points[bestLabels[i]].push_back(edge_nodes[i]);
		}
		int idx[2] = { -1,-1 };
		float max_dist = 0;
		for (int i = 0; i < num_labels - 1; i++) {
			for (int j = i + 1; j < num_labels; j++) {
				Point2f average_p1 = Point2f(sum_point[i].x / category_points[i].size(), sum_point[i].y / category_points[i].size());
				Point2f average_p2 = Point2f(sum_point[j].x / category_points[j].size(), sum_point[j].y / category_points[j].size());

				float dist = distance(average_p1, average_p2);
				if (dist > max_dist) {
					max_dist = dist;
					idx[0] = i;
					idx[1] = j;
				}
			}
		}
		Point p[2];
		for (int i = 0; i < 2; i++) {
			Point2f average_p = Point2f(sum_point[idx[i]].x / category_points[idx[i]].size(),
				sum_point[idx[i]].y / category_points[idx[i]].size());

			float minDist = numeric_limits<float>::max();
			int closet = 0;
			for (int j = 0; j < category_points[idx[i]].size(); j++) {
				float dist = distance(average_p, category_points[idx[i]][j]);
				if (dist < minDist) {
					minDist = dist;
					closet = j;
				}
			}
			p[i] = category_points[idx[i]][closet];
		}


		if (p[0].y > p[1].y) {
			start = p[1];
			end = p[0];
		}
		start = p[0];
		end = p[1];
	}
	else {
		Point rt(I1.cols, 0), lb(0, I1.rows);
		float min_rt = numeric_limits<float>::max();
		float min_lb = min_rt;
		for (int i = 0; i < edges.size(); i++) {
			if (distance(edges[i], rt) < min_rt) {
				start = edges[i];
				min_rt = distance(edges[i], rt);
			}

			if (distance(edges[i], lb) < min_lb) {
				end = edges[i];
				min_lb = distance(edges[i], lb);
			}
		}
	}

}

bool OptimalSeam::neighborhood_class(Point point, int class_num)
{
	set<int> class_set;
	int x = point.x, y = point.y;
	vector<Point2i> neighbors = { Point(x - 1, y - 1), Point(x - 1, y), Point(x - 1, y + 1), Point(x, y - 1),
		Point(x, y + 1), Point(x + 1, y - 1), Point(x + 1, y), Point(x + 1, y + 1) };
	for (int i = 0; i < neighbors.size(); i++)
	{
		if (labels(neighbors[i].y, neighbors[i].x) != 4) class_set.insert(labels(neighbors[i].y, neighbors[i].x));
	}

	if (class_set.size() >= class_num) return true;

	return false;
}

int OptimalSeam::minCostIndex(float left, float mid, float right, int correct_index)
{
	if (left <= right && left <= mid)
		return correct_index - 1;
	if (right <= left && right <= mid)
		return correct_index + 1;
	return correct_index;
}

float OptimalSeam::diffL2Square3(Mat img1, int x1, int y1, Mat img2, int x2, int y2)
{
	float* p1 = img1.ptr<float>(y1);
	float* p2 = img2.ptr<float>(y2);
	return square(p1[3 * x1] - p2[3 * x2]) + square(p1[3 * x1 + 1] - p2[3 * x2 + 1])
		+ square(p1[3 * x1 + 2] - p2[3 * x2 + 2]);
}



