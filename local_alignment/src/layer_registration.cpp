#include"layer_registration.h"
#include"feature_finder.h"

#include"warper.h"
#include"image_stitch.h"

#include"my_util.h"


vector<Mat> LayerRegistration::layer_homography(Mat src, Mat dst, int K, bool is_show)
{
	FeatureFinder finder;
	vector<vector<Point2f>> match_pair = finder.match_points(src, dst, false);
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 0.5);
	Mat labels, centers;
	int attempts = 150;
	kmeans(match_pair[0], K, labels, criteria, attempts, KMEANS_RANDOM_CENTERS, centers); 

	vector<int> labels_ = (vector<int>)labels;
	if (is_show) {
		vector<Scalar> colors;
		for (int i = 0; i < K; i++) {
			RNG rng(getTickCount());
			Scalar c = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			colors.push_back(c);
		}
		Mat canvas(src.rows, 2 * src.cols, CV_8UC3);
		dst.copyTo(canvas(Rect(0, 0, dst.cols, dst.rows)));
		src.copyTo(canvas(Rect(src.cols, 0, src.cols, src.rows)));
		for (int i = 0; i < labels_.size(); i++) {
			circle(canvas, match_pair[0][i] + Point2f(src.cols, 0), 5, colors[labels_[i]], -1);
			circle(canvas, match_pair[1][i], 5, colors[labels_[i]], -1);
		}
		
		MyUtil::showImg("layer_feature_match", canvas);
	}

	classify_s = vector<vector<Point2f>>(K), classify_d = vector<vector<Point2f>>(K);
	for (int i = 0; i < labels_.size(); i++) { 
		classify_s[labels_[i]].push_back(match_pair[0][i]);
		classify_d[labels_[i]].push_back(match_pair[1][i]);
	}
	vector<Mat> layerHs;
	int Nmin = 12;
	for (int i = 0; i < K; i++) {
		vector<uchar> mask;
		Mat H = findHomography(classify_s[i], classify_d[i], mask, RANSAC, 3);


		int inliers = count(mask.begin(), mask.end(), 1);
		int outliers = count(mask.begin(), mask.end(), 0);
		if (inliers >= Nmin) { 
			cout << "classify " << i << ": \n" << H << endl;
			layerHs.push_back(H);
		}
		else {
			layerHs.push_back(Mat());
			cout << "The match points of classify " << i << ": " << inliers << " is less than 12." << endl;
		}
	}
	return layerHs;
}


bool LayerRegistration::grid_homography(Mat src, Mat dst, int K, Size grid)
{
	vector<Mat> layerHs = layer_homography(src, dst, K, false); 
	
	int count_layerHs = 0;
	for (int k = 0; k < K; k++)
		if (!layerHs[k].empty())  count_layerHs++;		
	if (count_layerHs < 2) {
		cout << "The homography layers "<< count_layerHs <<" is less than 2, and the layer features is not enough!" << endl;
		return false;
	}

	int cell_width = dst.cols / grid.width, cell_height = dst.rows / grid.height; 

	grid_rows = ceil(float(dst.rows) / cell_height), grid_cols = ceil(float(dst.cols) / cell_width); 

	if (grid_Hs != NULL) {
		for (int i = 0; i < grid_rows; i++)
			delete[] grid_Hs[i]; 
		delete[] grid_Hs;
		grid_Hs = NULL;
	}
	grid_Hs = new Mat_<float> * [grid_rows];
	for (int i = 0; i < grid_rows; i++)
		grid_Hs[i] = new Mat_<float>[grid_cols];

	for (int y = 0; y < grid_rows; y++) {
		for (int x = 0; x < grid_cols; x++) {
			float c_xx = x * cell_width + float(cell_width) / 2, c_yy = y * cell_height + float(cell_height) / 2;
			if ((x + 1) * cell_width > dst.cols)
				c_xx = float(x * cell_width + dst.cols) / 2;
			if ((y + 1) * cell_height > dst.rows)
				c_yy = float(y * cell_height + dst.rows) / 2;
			Point2f center = Point2f(c_xx, c_yy);

			vector<float> a_ks;
			vector<Mat> layerHs_; 
			float sum = 0;
			for (int k = 0; k < K; k++) {
				float a_k = nearest_gaussian_weight(center, classify_d[k]); 
				if (!layerHs[k].empty()) {
					layerHs_.push_back(layerHs[k]);
					a_ks.push_back(a_k);
					sum += a_k;
				}
			}
			Mat H_ij = Mat::zeros(3, 3, CV_32F);
			for (int k = 0; k < a_ks.size(); k++) { 
				layerHs_[k].convertTo(layerHs_[k], CV_32F);
				H_ij += a_ks[k] / sum * layerHs_[k]; 
			}

			grid_Hs[y][x] = H_ij;
		}
	}
	return true;
}


double LayerRegistration::nearest_gaussian_weight(Point2f center, vector<Point2f> class_points)
{
	double min_distance = MyUtil::euclid(center, class_points[0]);
	Point2f nearest_point = class_points[0];
	for (int i = 1; i < class_points.size(); i++) {
		double distance = MyUtil::euclid(center, class_points[i]);
		if (distance < min_distance) {
			min_distance = distance;
			nearest_point = class_points[i];
		}
	}
	double sigma = 5.0;
	double a_k = exp(-min_distance / pow(sigma, 2));

	return a_k;
}


void LayerRegistration::grid_perspective(Mat src, Mat& warp)
{
	perspectiveCorner(src);
	Size size(corner.pmax.x, corner.pmax.y);
	warp = Mat::zeros(size, CV_8UC3);

	Mat map = -1 * Mat::ones(size, CV_32FC2);

	int cell_width = ceil(float(src.cols) / grid_cols), cell_height = ceil(float(src.rows) / grid_rows);
	for (int i = 0; i < src.rows; i += cell_height) {
		for (int j = 0; j < src.cols; j += cell_width) {
			int y = i / cell_height, x = j / cell_width;
			Mat H_xy = grid_Hs[y][x];
			build_cell_map(j, i, min(j + cell_width, src.cols), min(i + cell_height, src.rows), H_xy, map);
		}
	}

	Mat maps[2];
	split(map, maps);

	remap(src, warp, maps[0], maps[1], INTER_LINEAR);
}


void LayerRegistration::build_cell_map(int x_start, int y_start, int x_end, int y_end, Mat H, Mat& map)
{
	H.convertTo(H, CV_32F);

	ImageStitch is;
	vector<Point2f> warp_corners, four_corners = { Point2f(x_start, y_start), Point2f(x_end, y_start), Point2f(x_end, y_end), Point2f(x_start, y_end) };
	Point2f p = is.warpPoint(four_corners[0], H);
	warp_corners.push_back(p);
	MMCorner mm(p, p);
	for (int i = 1; i < four_corners.size(); i++) {
		p = is.warpPoint(four_corners[i], H);
		mm.pmin.x = (p.x < mm.pmin.x ? p.x : mm.pmin.x);
		mm.pmin.y = (p.y < mm.pmin.y ? p.y : mm.pmin.y);
		mm.pmax.x = (p.x > mm.pmax.x ? p.x : mm.pmax.x);
		mm.pmax.y = (p.y > mm.pmax.y ? p.y : mm.pmax.y);
		warp_corners.push_back(p);
	}

	int y_min = floor(mm.pmin.y), y_max = ceil(mm.pmax.y);
	int x_min = floor(mm.pmin.x), x_max = ceil(mm.pmax.x);

	if (x_min > 0 && x_max < map.cols && y_min>0 && y_max < map.rows) { 
		for (int y = y_min; y < y_max; y++) {
			float* map_ptr = map.ptr<float>(y);
			for (int x = x_min; x <= x_max; x++) {
				p = is.warpPoint(Point2f(x, y), H.inv());  
				map_ptr[2 * x] = p.x;
				map_ptr[2 * x + 1] = p.y;
			}
		}

	}
}


void LayerRegistration::perspectiveCorner(Mat img)
{
	ImageStitch is;
	Point2f p = is.warpPoint(Point2f(0, 0), grid_Hs[0][0]);
	corner.pmin = p, corner.pmax = p;

	int cell_width = ceil(float(img.cols) / grid_cols), cell_height = ceil(float(img.rows) / grid_rows);

	for (int x = 0; x < img.cols; x += cell_width) {
		updateMMCorner(grid_Hs[0][x / cell_width], x, 0); 
		updateMMCorner(grid_Hs[grid_rows - 1][x / cell_width], x, img.rows - 1); 
	}
	for (int y = 0; y < img.rows; y += cell_height) {
		updateMMCorner(grid_Hs[y / cell_height][0], 0, y); 
		updateMMCorner(grid_Hs[y / cell_height][grid_cols - 1], img.cols - 1, y); 
	}
}

void LayerRegistration::updateMMCorner(Mat H, float x, float y)
{
	ImageStitch is;

	Point2f p = is.warpPoint(Point2f(x, y), H);

	corner.pmax.x = (p.x > corner.pmax.x ? p.x : corner.pmax.x);
	corner.pmax.y = (p.y > corner.pmax.y ? p.y : corner.pmax.y);
	corner.pmin.x = (p.x < corner.pmin.y ? p.x : corner.pmin.x);
	corner.pmin.y = (p.y < corner.pmin.y ? p.y : corner.pmin.y);
}



void LayerRegistration::test_fun()
{
	Mat img1 = imread("../multi_video_stitching/images/001.jpg");
	Mat img2 = imread("../multi_video_stitching/images/002.jpg");


	ImageStitch is;
	is.cylindrical = true;
	Mat pano = is.direct_stitch(img2, img1);
	MyUtil::showImg("homography_pano", pano);


	double t = getTickCount();
	Warper w;
	img1 = w.cylindrical_projection(img1);
	img2 = w.cylindrical_projection(img2);
	bool isH = grid_homography(img2, img1, 3, Size(300, 260)); 
	Mat warp;
	if (isH) {
		grid_perspective(img2, warp);  
		Mat pano2 = is.average_weight(img1, warp, corner.pmin);
		MyUtil::showImg("layer_pano", pano2);
	}
	cout << "perspective elapsed time: " << (getTickCount() - t) / getTickFrequency() << "s" << endl;

	waitKey();
}

void LayerRegistration::test_layerH()
{
	Mat dst = imread("../multi_video_stitching/images/001.jpg");
	Mat src = imread("../multi_video_stitching/images/002.jpg");
	Warper w;
	dst = w.cylindrical_projection(dst);
	src = w.cylindrical_projection(src);
	vector<Mat> layerHs = layer_homography(src, dst, 5, true); 
	
	for (int i = 0; i < layerHs.size(); i++) {
		if (layerHs[i].empty()) continue;
		ImageStitch is;
		Point2f* ps = is.warpSize(src, layerHs[i]);
		Mat warp;
		warpPerspective(src, warp, layerHs[i], Size(ps[1].x, ps[1].y));
		Mat pano = is.average_weight(dst, warp, ps[0]);
		MyUtil::showImg("pano" + to_string(i), pano);
	}
	waitKey();
}

