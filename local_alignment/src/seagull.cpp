#include"seagull.h"
#include"my_util.h"
#include"image_stitch.h"
#include"optimal_seam.h"
#include"group_feature.h"

#include<ximgproc\fast_line_detector.hpp>



vector<Point2f> SEAGULL::merge_vec(vector<vector<Point2f>> vecs)
{
	vector<Point2f> vec;
	if (vecs.size() == 0) {
		vec = vecs[0];
	}
	else {
		for (int i = 1; i < vecs.size(); i++) {
			vec.insert(vec.end(), vecs[i].begin(), vecs[i].end());
		}
	}

	return vec;
}

float SEAGULL::shortest_dist(vector<Point2f> vec1, vector<Point2f> vec2)
{
	float min_dist = numeric_limits<float>::max();
	for (int i = 0; i < vec1.size(); i++) {
		for (int j = 0; j < vec2.size(); j++) {
			float dist = MyUtil::euclid(vec1[i], vec2[j]);
			if (min_dist > dist) min_dist = dist;
		}
	}
	return min_dist;
}




vector<Point2f> SEAGULL::seam_points(Mat src, Mat dst, Mat H)
{
	ImageStitch is;
	Point2f* ps = is.warpSize(src, H);
	Mat warp;
	warpPerspective(src, warp, H, Size(ps[1].x, ps[1].y));
	Mat dst_ = Mat::zeros(max(warp.rows, dst.rows), dst.cols, CV_8UC3), warp_ = dst_.clone();
	dst.copyTo(dst_(Rect(0, 0, dst.cols, dst.rows)));
	warp.copyTo(warp_(Rect(0, 0, warp.cols, warp.rows)));
	Mat cut1 = dst_(Rect(ps[0].x, 0, dst_.cols - ps[0].x, dst_.rows));
	Mat cut2 = warp_(Rect(ps[0].x, 0, dst_.cols - ps[0].x, warp_.rows));
	OptimalSeam os;
	os.DP_find_seam(cut1, cut2);

	vector<Point2f> seam;
	for (int i = 0; i < cut2.rows; i++) {
		seam.push_back(Point2f(os.seam_paths[i], i));
	}
	return seam;
}

void SEAGULL::grid_image(Mat src)
{
	int step_x = ceil(float(src.cols) / grid.width), step_y = ceil(float(src.rows) / grid.height);
	grid_label_matrix = Mat::zeros(src.size(), CV_16UC1);
	int tmp_label = 0;
	for (int y = 0; y < src.rows; y += step_y) {
		for (int x = 0; x < src.cols; x += step_x) {
			int cell_width = step_x, cell_height = step_y;
			if (y + step_y > src.rows) cell_height = src.rows - y;
			if (x + step_x > src.cols) cell_width = src.cols - x;
			Mat cell_label = tmp_label * Mat::ones(cell_height, cell_width, CV_16UC1);
			cell_label.copyTo(grid_label_matrix(Rect(x, y, cell_width, cell_height)));

			cells.push_back(Rect2f(x, y, cell_width, cell_height));
			tmp_label++;
		}
	}
}


float SEAGULL::feature_term(vector<Point2f> group_s, vector<Point2f> group_d, float ds)
{
	Mat H = findHomography(group_s, group_d, RANSAC);

	float lambda;
	if (ds <= 20) lambda = 1.5;
	else lambda = 0.1;

	ImageStitch is;
	float Ef_V = 0;
	for (int i = 0; i < group_s.size(); i++) {
		Point2f warp_point = is.warpPoint(group_s[i], H);
		float dm = MyUtil::euclid(warp_point, group_d[i]);
		float wi = lambda * (exp(-dm * dm / (2 * 10 * 10)) + 0.01);
		Rect2f rect = cells[grid_label_matrix(group_s[i].y, group_s[i].x)];
		Point2f mul_sum = sum_ck_Vk(rect, group_s[i], H);
		Ef_V += (wi * MyUtil::L2_square(mul_sum, group_d[i]));
	}
	return Ef_V;
}

Point2f SEAGULL::sum_ck_Vk(Rect2f rect, Point2f p, Mat H)
{
	ImageStitch is;
	Point2f Vk1_ = is.warpPoint(Point2f(rect.x, rect.y), H);
	Point2f Vk2_ = is.warpPoint(Point2f(rect.x + rect.width, rect.y), H);
	Point2f Vk3_ = is.warpPoint(Point2f(rect.x, rect.y + rect.height), H);
	Point2f Vk4_ = is.warpPoint(Point2f(rect.x + rect.width, rect.y + rect.height), H);
	float u = abs(p.x - rect.x)/rect.width, v = abs(p.y - rect.y)/rect.height;

	return (1 - u) * (1 - v) * Vk1_ + (1 - v) * u * Vk2_ + (1 - u) * v * Vk3_ + u * v * Vk4_;
}


float SEAGULL::local_similarity_term(Mat H)
{
	float Els = 0;
	for (int i = 0; i < cells.size(); i++) {
		Rect2f rect = cells[i];		
		Els += cell_error(rect, H);
	}
	return Els;
}

float SEAGULL::cell_error(Rect2f rect, Mat H)
{
	Point2f p1(rect.x, rect.y), p2(rect.x, rect.y + rect.height), p3(rect.x + rect.width, rect.y), p4(rect.x + rect.width, rect.y + rect.height);
	Point2f vec21 = p1 - p2, vec23 = p3 - p2, vec24 = p4 - p2; 
	float u1 = (vec23.x * vec21.x + vec23.y * vec21.y) / (vec23.x * vec23.x + vec23.y * vec23.y); 
	float u2 = (vec23.x * vec24.x + vec23.y * vec24.y) / (vec23.x * vec23.x + vec23.y * vec23.y);
	Point2f vec23_R90 = Point2f(-vec23.y, vec23.x); 
	float v1 = (vec23_R90.x * vec21.x + vec23_R90.y * vec21.y) / (vec23_R90.x * vec23_R90.x + vec23_R90.y * vec23_R90.y); 
	float v2 = (vec23_R90.x * vec24.x + vec23_R90.y * vec24.y) / (vec23_R90.x * vec23_R90.x + vec23_R90.y * vec23_R90.y);

	ImageStitch is;
	Point2f Vk1_ = is.warpPoint(p1, H);
	Point2f Vk2_ = is.warpPoint(p2, H);
	Point2f Vk3_ = is.warpPoint(p3, H);
	Point2f Vk4_ = is.warpPoint(p4, H);

	Point2f Vk23 = Vk3_ - Vk2_;
	Point2f delta1 = Vk1_ - (Vk2_ + u1 * Vk23 + v1 * Point2f(-Vk23.y, Vk23.x));
	float Ctri = delta1.x * delta1.x + delta1.y * delta1.y;
	Point2f delta2 = Vk4_ - (Vk2_ + u2 * Vk23 + v2 * Point2f(-Vk23.y, Vk23.x));
	Ctri += delta2.x * delta2.x + delta2.y * delta2.y;

	return Ctri;
}

float SEAGULL::line_error(Vec4f line, vector<Point2f> keys, Mat H)
{
	float Ecs_line = 0;
	for (int i = 0; i < keys.size(); i++) {
		Point2f Vb(line[0], line[1]), Vc(line[2], line[3]), Va=keys[i];
		Point2f vec_bc = Vc - Vb, vec_ba = Va - Vb; 
		float u = (vec_bc.x * vec_ba.x + vec_bc.y * vec_ba.y) / (vec_bc.x * vec_bc.x + vec_bc.y * vec_bc.y); 
		Point2f vec_bcR90 = Point2f(-vec_bc.y, vec_bc.x); 
		float v = (vec_bcR90.x * vec_ba.x + vec_bcR90.y * vec_ba.y) / (vec_bcR90.x * vec_bcR90.x + vec_bcR90.y * vec_bcR90.y); 

		ImageStitch is;
		Point2f Va_ = is.warpPoint(Va, H);
		Point2f Vb_ = is.warpPoint(Vb, H);
		Point2f Vc_ = is.warpPoint(Vc, H);

		Point2f vec_bc_ = Vc_ - Vb_;
		Point2f delta = Va_ - (Vb_ + u * vec_bc_ + v * Point2f(-vec_bc_.y, vec_bc_.x));

		Ecs_line += delta.x * delta.x + delta.y * delta.y;
	}
	return Ecs_line;
}

float SEAGULL::non_local_similarity_term(Mat src, vector<Point2f> keys, Mat H)
{
	vector<Vec4f> lines;
	Mat src_;
	Canny(src, src_, 150, 250, 3);
	Ptr<ximgproc::FastLineDetector> fld = ximgproc::createFastLineDetector();
	fld->detect(src_, lines);

	float Ecs = 0;
	for (int i = 0; i < lines.size(); i++) {
		if (abs(lines[i][0] - lines[i][2]) > 30 || abs(lines[i][1] - lines[i][3]) > 30) {
			Ecs += line_error(lines[i], keys, H);
		}
	}

	return Ecs;
}


void SEAGULL::seam_iteration(Mat src, Mat dst)
{
	grid_image(src);

	GroupFeature gf;
	gf.kmean_group(src, dst, 5, false);
	for (int i = 0; i < gf.groups_src.size(); i++) {
		vector<Point2f> src_points = gf.groups_src[i], dst_points = gf.groups_dst[i];
		Mat H = findHomography(src_points, dst_points, RANSAC);
		cout << "H:" << H << endl;
		
		float Ef = feature_term(src_points, dst_points, 0); 
		float Els = local_similarity_term(H);
		float Ecs = non_local_similarity_term(src, src_points, H);
		float E_sum = 5 * Ef + Els + 10 * Ecs;
		cout << E_sum << endl;
	}
}


void SEAGULL::test()
{
	Mat img1 = imread("../multi_video_stitching/images/temple1.jpg");
	Mat img2 = imread("../multi_video_stitching/images/temple2.jpg");

	seam_iteration(img2, img1);
}
