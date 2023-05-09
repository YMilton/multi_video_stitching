#include"group_feature.h"

#include"feature_finder.h"
#include"my_util.h"

#include<ximgproc\slic.hpp>


bool GroupFeature::is_neighbor(vector<int> vec_labels, int label, Mat_<int> label_matrix)
{
	for (int y = 0; y < label_matrix.rows; y++) {
		for (int x = 0; x < label_matrix.cols; x++) {
			for (int k = 0; k < vec_labels.size(); k++) {
				int label2 = vec_labels[k];
				if (label_matrix(y, x) == label && ((x < label_matrix.cols - 1 && label_matrix(y, x + 1) == label2)
					|| (x > 0 && label_matrix(y, x - 1) == label2)
					|| (y < label_matrix.rows - 1 && label_matrix(y + 1, x) == label2)
					|| (y > 0 && label_matrix(y - 1, x) == label2))) return true;
			}
		}
	}

	return false;
}

bool GroupFeature::is_neighbor(int label1, int label2, Mat_<int> label_matrix)
{
	for (int y = 0; y < label_matrix.rows; y++) {
		for (int x = 0; x < label_matrix.cols; x++) {
			if (label_matrix(y, x) == label1 && ((x < label_matrix.cols - 1 && label_matrix(y, x + 1) == label2)
				|| (x > 0 && label_matrix(y, x - 1) == label2)
				|| (y < label_matrix.rows - 1 && label_matrix(y + 1, x) == label2)
				|| (y > 0 && label_matrix(y - 1, x) == label2))) return true;
		}
	}
	return false;
}




void GroupFeature::kmean_group(Mat src, Mat dst, int K, bool is_show)
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

	vector<vector<Point2f>> classify_s = vector<vector<Point2f>>(K), classify_d = vector<vector<Point2f>>(K);
	for (int i = 0; i < labels_.size(); i++) { 
		classify_s[labels_[i]].push_back(match_pair[0][i]);
		classify_d[labels_[i]].push_back(match_pair[1][i]);
	}
	
	int Nmin = 12; 
	groups_src.clear(), groups_dst.clear();
	for (int i = 0; i < K; i++) {
		vector<uchar> mask;
		Mat H = findHomography(classify_s[i], classify_d[i], mask, RANSAC, 3);
		vector<Point2f> screen_src, screen_dst;
		for (int j = 0; j < classify_s[i].size(); j++) {
			if (mask[j]) {
				screen_src.push_back(classify_s[i][j]);
				screen_dst.push_back(classify_d[i][j]);
			}
		}

		if (screen_src.size() >= Nmin) { 
			groups_src.push_back(screen_src);
			groups_dst.push_back(screen_dst);
		}
	}
}

void GroupFeature::superpixel_group(Mat src, Mat dst, bool is_show)
{
	SuperPixelFeature spf = features_into_superpixel(src, dst, is_show);
	Mat_<int> label_matrix = spf.label_matrix;
	vector<int> labels = spf.labels;
	vector<vector<Point2f>> classify_s = spf.classify_s, classify_d = spf.classify_d;

	if (labels.size() == 0) {
		cout << "exetute the function features_into_superpixel firstly." << endl;
		return;
	}

	vector<pair<int, int>> feature_count; 
	for (int i = 0; i < classify_s.size(); i++) {
		feature_count.push_back(make_pair(i, classify_s[i].size()));
	}
	sort(feature_count.begin(), feature_count.end(), my_comp); 

	vector<vector<Point2f>> merge_s, merge_d;
	for (; feature_count.size() != 0; ) {
		int idx = feature_count[0].first;
		cout << "max_feature label " << labels[idx] << "," << feature_count[0].second << ":" << endl;
		merge_s.push_back(classify_s[idx]); merge_d.push_back(classify_d[idx]);
		vector<int> vec_labels{ labels[idx] }; 

		feature_count.erase(feature_count.begin());
		int i = 0; bool is_add = false;
		for (; i < feature_count.size(); ) {
			if (is_add) {
				feature_count.erase(feature_count.begin() + i); 
				if (feature_count.size() == 0) break; 
				i = 0; 
			}
			else {
				i++;
			}

			if (is_neighbor(vec_labels, labels[feature_count[i].first], label_matrix)) {
				cout << labels[feature_count[i].first] << " ";
				vector<Point2f> tmp_vec_s = merge_s[merge_s.size() - 1], tmp_vec_d = merge_d[merge_d.size() - 1];
				tmp_vec_s.insert(tmp_vec_s.end(), classify_s[feature_count[i].first].begin(), classify_s[feature_count[i].first].end());
				tmp_vec_d.insert(tmp_vec_d.end(), classify_d[feature_count[i].first].begin(), classify_d[feature_count[i].first].end());

				vector<uchar> tmp_mask;
				Mat H = findHomography(tmp_vec_s, tmp_vec_d, tmp_mask, RANSAC, 5.0);
				if (!H.empty()) { 
					merge_s[merge_s.size() - 1] = tmp_vec_s;
					merge_d[merge_s.size() - 1] = tmp_vec_d;
					vec_labels.push_back(labels[feature_count[i].first]); 
					is_add = true; 
				}
			}
			else {
				is_add = false;
			}
		}
		cout << endl;
	}

	groups_src = merge_s; groups_dst = merge_d;
}

SuperPixelFeature GroupFeature::features_into_superpixel(Mat src, Mat dst, bool is_show)
{
	FeatureFinder finder;
	vector<vector<Point2f>> match_points = finder.match_points(src, dst, false);
	vector<uchar> mask;
	Mat H = findHomography(match_points[0], match_points[1], mask, RANSAC);

	SuperPixel sp = superpixel_slic(src); 
	Mat_<int> label_matrix = sp.lables;
	vector<int> labels;
	vector<vector<Point2f>> classify_s, classify_d;
	for (int i = 0; i < mask.size(); i++) {
		int element = sp.lables(match_points[0][i]);
		if (labels.size() != 0 && count(labels.begin(), labels.end(), element)) { 
			vector<int>::iterator it = find(labels.begin(), labels.end(), element); 
			int idx = distance(labels.begin(), it); 
			classify_s[idx].push_back(match_points[0][i]);
			classify_d[idx].push_back(match_points[1][i]);
		}
		else {
			labels.push_back(element); 
			classify_s.push_back(vector<Point2f>{match_points[0][i]});
			classify_d.push_back(vector<Point2f>{match_points[1][i]});
		}
	}

	if (is_show) {
		for (int i = 0; i < classify_s.size(); i++) {
			RNG rng(getTickCount());
			Scalar c = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			for (int k = 0; k < classify_s[i].size(); k++) {
				circle(sp.sp_img, classify_s[i][k], 3, c, -1);
			}
			putText(sp.sp_img, to_string(labels[i]), classify_s[i][0], FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 0, 255));
		}
		MyUtil::showImg("slic_src", sp.sp_img);
	}

	SuperPixelFeature spf;
	spf.label_matrix = label_matrix;
	spf.labels = labels;
	spf.classify_d = classify_d;
	spf.classify_s = classify_s;

	return spf;
}


SuperPixel GroupFeature::superpixel_slic(Mat img)
{
	SuperPixel sp;
	Mat mask, labels;
	int region_size = 50; 

	Ptr<ximgproc::SuperpixelSLIC> slic = ximgproc::createSuperpixelSLIC(img, ximgproc::SLICO, region_size);
	slic->iterate(15); 
	slic->enforceLabelConnectivity();
	slic->getLabels(labels); 
	slic->getLabelContourMask(mask); 

	int num = slic->getNumberOfSuperpixels(); 
	Mat img_ = img.clone();
	img_.setTo(Scalar(255, 255, 255), mask);

	sp.lables = labels;
	sp.mask = mask;
	sp.num = num;
	sp.sp_img = img_;

	return sp;
}

void GroupFeature::test()
{
	Mat img1 = imread("../multi_video_stitching/images/temple1.jpg");
	Mat img2 = imread("../multi_video_stitching/images/temple2.jpg");


	superpixel_group(img2, img1, false);

	waitKey();
}
