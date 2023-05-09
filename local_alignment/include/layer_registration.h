/*
  Paper: Parallax-Robust Surveillance Video Stitching
         Constructing Image Panoramas using Dual-Homography Warping
  Author: YMilton

  Using K-Means to group keypoints and calculate layer homography,
  then local alignment through layer homography.
*/
#pragma once
#include<opencv.hpp>

using namespace cv;
using namespace std;

class MMCorner {
public:
    Point2f pmin, pmax;

    MMCorner() {}
    MMCorner(Point pmin, Point pmax) { this->pmin = pmin; this->pmax = pmax; }
};

class LayerRegistration
{
private:
    vector<vector<Point2f>> classify_s, classify_d;

public:

    Mat_<float>** grid_Hs;
    int grid_rows, grid_cols;

    MMCorner corner; 

    vector<Mat> layer_homography(Mat src, Mat dst, int K, bool is_show);


    bool grid_homography(Mat src, Mat dst, int K, Size grid);


    double nearest_gaussian_weight(Point2f center, vector<Point2f> class_points);


    void grid_perspective(Mat src, Mat& warp);


    void build_cell_map(int x_start, int y_start, int x_end, int y_end, Mat H, Mat& map);


    void perspectiveCorner(Mat img);

    void updateMMCorner(Mat H, float x, float y);


    void test_fun();

    void test_layerH();
};