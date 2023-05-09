#pragma once
#include<opencv.hpp>

using namespace cv;
using namespace std;

class Camera
{
public:
	Size resolution = Size(640,480); 

	bool initCamera(VideoCapture* cap, string num_or_path);

	bool grab_cameras(vector<VideoCapture*> caps);
	vector<Mat> getFrames(vector<VideoCapture*> caps);

	void save_mutil_video(vector<string> cameras,string path_name);
	void cut_videos(vector<string> videos, string path_name, int begin, int end=-1);

	void first_frames(vector<string> num_or_path, string savePath);
	void save1Frame(vector<string> num_or_path, string save_path);

	string get_current_time();

	void view_frame_detail(string path);

	void output_camera_args(VideoCapture* cap);
};

