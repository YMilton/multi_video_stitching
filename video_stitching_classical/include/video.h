#pragma once
#include<opencv.hpp>

using namespace cv;
using namespace std;

class Video
{
public:
    vector<Mat> cut_frame(Mat frame, Size grid, vector<int> order);


    void test_video();
};

