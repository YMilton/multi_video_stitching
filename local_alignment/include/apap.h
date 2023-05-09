#ifndef __ZFD_APAP_H__
#define __ZFD_APAP_H__

#include<opencv2/stitching/detail/warpers.hpp>
#include<opencv2/stitching/detail/matchers.hpp>
#include<vector>

using namespace cv;
using namespace std;
using namespace cv::detail;

typedef double apap_float;

class APAPWarper {

public:
	APAPWarper();
	~APAPWarper();
	int warp(Mat src_img, ImageFeatures src_features, ImageFeatures dst_features,
		MatchesInfo matches_info, Mat& result_img, Point& corner);
	int buildMaps(Mat src_img, ImageFeatures src_features, ImageFeatures dst_features,
		MatchesInfo matches_info, Mat& xmap, Mat& ymap, Point& corner);
	int buildMaps(vector<Mat> imgs, vector<ImageFeatures> features,
		vector<MatchesInfo> pairwise_matches, vector<Mat>& xmaps, vector<Mat>& ymaps, vector<Point>& corners);

protected:

private:
	int CellDLT(int offset_x, int offset_y, ImageFeatures src_features,
		ImageFeatures dst_features, MatchesInfo matches_info, Mat_<apap_float>& H);
	void InitA(ImageFeatures src_features, ImageFeatures dst_features,
		MatchesInfo matches_info);
	void testh(ImageFeatures src_features, ImageFeatures dst_features,
		MatchesInfo matches_info, Mat h);

	void BuildCellMap(int x1, int y1, int x2, int y2, Mat H, Point corner, Mat& xmap, Mat& ymap);

	inline void updateMappedSize(Mat_<apap_float> H, double x, double y,
		Point2d& corner, Point2d& br);

	int cell_height_;
	int cell_width_;
	double gamma_;
	double sigma_;

	Mat_<apap_float> A_;
	Mat_<apap_float> W_;

	Mat_<apap_float>** H_s_;
	int cell_rows_, cell_cols_;

};


#endif
