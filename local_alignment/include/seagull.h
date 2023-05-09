/*
 Paper: Kaimo Lin, Nianjuan Jiang etc,SEAGULL: Seam-guided Local Alignment for Parallax-tolerant Image Stitching
 Author: YMilton

 1.grouping the feature points in superpixels by iteration.
 2.get the seam by feature term and structure-preserving terms.
*/
#pragma once
#include<opencv.hpp>

using namespace cv;
using namespace std;



class SEAGULL
{
private:
    vector<Point2f> merge_vec(vector<vector<Point2f>> vecs);

    float shortest_dist(vector<Point2f> vec1, vector<Point2f> vec2);



public:
    Size grid = Size(60, 45);
    vector<Rect2f> cells; 
    Mat_<int> grid_label_matrix; 


    vector<Point2f> seam_points(Mat src, Mat dst, Mat H);
    void grid_image(Mat src);
   

    float feature_term(vector<Point2f> group_s, vector<Point2f> group_d, float ds);
    Point2f sum_ck_Vk(Rect2f rect, Point2f p, Mat H);

    float local_similarity_term(Mat H);
    float cell_error(Rect2f rect, Mat H);
    float line_error(Vec4f line, vector<Point2f> keys, Mat H);
    float non_local_similarity_term(Mat src, vector<Point2f> keys, Mat H);


    
    void seam_iteration(Mat src, Mat dst);

    void test();
};

