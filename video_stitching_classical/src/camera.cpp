#include"camera.h"
#include"my_util.h"
#include<ctime>
#include<chrono>

bool Camera::initCamera(VideoCapture* cap, string num_or_path)
{
	bool isOpen;
	if (num_or_path.length() == 1) { 
		isOpen = cap->open(stoi(num_or_path));
		cap->set(CAP_PROP_FRAME_WIDTH, resolution.width);
		cap->set(CAP_PROP_FRAME_HEIGHT, resolution.height);
	}
	else { 
		isOpen = cap->open(num_or_path);
	}
	 
	if (!isOpen) {
		cout << num_or_path+" open camera or video abnormal!" << endl;
		return false;
	}

	return true;
}


bool Camera::grab_cameras(vector<VideoCapture*> caps)
{
	bool flag = true;
	for (int i = 0; i < caps.size(); i++) {
		flag = flag && caps[i]->grab();
		if (!flag) break;
	}
	return flag;
}

vector<Mat> Camera::getFrames(vector<VideoCapture*> caps)
{
	vector<Mat> frames;
	for (int i = 0; i < caps.size(); i++) {
		Mat frame;
		caps[i]->retrieve(frame);
		frames.push_back(frame.clone());
	}
	return frames;
}

void Camera::save_mutil_video(vector<string> cameras, string path_name)
{
	MyUtil::createDir(path_name);
	vector<VideoCapture*> caps;
	for (int i = 0; i < cameras.size(); i++) {
		VideoCapture* cap = new VideoCapture();
		resolution = Size(1280, 960); 
		bool is_init = initCamera(cap, cameras[i]);

		if (!is_init) {
			delete cap;
			continue;
		}
		else {
			cout << cameras[i] + " init success!" << endl;
		}
		caps.push_back(cap);
	}

	int f = 0;
	vector<VideoWriter*> writers;
	for (;;) {
		bool isFrame = grab_cameras(caps); 
		if (!isFrame) break;
		vector<Mat> frames = getFrames(caps);

		if (f == 0) {
			for (int k = 0; k < frames.size(); k++) {
				VideoWriter* vw = new VideoWriter();
				vw->open(path_name+"/camera" + to_string(k)+".avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.0,
					Size(frames[k].cols, frames[k].rows), true);
				writers.push_back(vw);
				*writers[k] << frames[k];
			}
		}else{
			for (int k = 0; k < frames.size(); k++) {
				*writers[k] << frames[k];
				MyUtil::showImg("camera" + to_string(k), frames[k]);
			}
		}

		if (waitKey(33) == 27) break;
		f++;
	}

	for (int i = 0; i < caps.size(); i++) {
		delete writers[i];
		delete caps[i];
	}
}

void Camera::cut_videos(vector<string> videos, string path_name, int begin, int end)
{
	MyUtil::createDir(path_name);
	vector<VideoCapture*> caps;
	for (int i = 0; i < videos.size(); i++) {
		VideoCapture* cap = new VideoCapture();
		bool is_init = initCamera(cap, videos[i]);

		if (!is_init) {
			delete cap;
			continue;
		}
		else {
			cout << videos[i] + " init success!" << endl;
		}
		caps.push_back(cap);
	}
	

	int f = 0;
	vector<VideoWriter*> writers;
	for (;;) {
		bool isFrame = grab_cameras(caps); 
		if (!isFrame) break;
		vector<Mat> frames = getFrames(caps);

		if (f == 0) {
			for (int k = 0; k < frames.size(); k++) {
				VideoWriter* vw = new VideoWriter();
				vw->open(path_name + "/camera" + to_string(k) + ".avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.0,
					Size(frames[k].cols, frames[k].rows), true);
				writers.push_back(vw);
			}
		}
		
		if (f >= begin && f <= end) {
			cout << "frame: " << f << endl;
			for (int k = 0; k < frames.size(); k++) {
				*writers[k] << frames[k];
				MyUtil::showImg("camera" + to_string(k), frames[k]);
			}
		}

		if (waitKey(33) == 27) break;
		f++;
	}

	for (int i = 0; i < caps.size(); i++) {
		delete writers[i];
		delete caps[i];
	}
}

void Camera::first_frames(vector<string> num_or_path, string savePath)
{
	MyUtil::createDir(savePath); 
	for (int i = 0; i < num_or_path.size(); i++) {
		VideoCapture* cap = new VideoCapture();
		resolution = Size(1280, 960);
		bool is_init = initCamera(cap, num_or_path[i]);

		if (!is_init) {
			delete cap;
			continue;
		}
		else {
			Mat frame;
			cap->read(frame);
			imwrite(savePath + "/camera" + to_string(i) + ".jpg", frame);
			cout << num_or_path[i] + " first frame save success!" << endl;
		}
	}
}

void Camera::save1Frame(vector<string> num_or_path, string save_path)
{
	MyUtil::createDir(save_path);
	vector<VideoCapture*> videos;
	for (int i = 0; i < num_or_path.size(); i++)
	{
		VideoCapture* vc = new VideoCapture();
		resolution = Size(1920, 1080);
		bool is_init = initCamera(vc, num_or_path[i]);
		if (!is_init) {
			delete vc;
			continue;
		}
		else {
			output_camera_args(vc);
			cout << num_or_path[i] + " init success!" << endl;
		}
		videos.push_back(vc);
	}

	for (;;){
		Mat frame;
		if (waitKey(10) == 115) {
			for (int i = 0; i < videos.size(); i++)
			{
				*videos[i] >> frame;
				imwrite(save_path + "/camera" + to_string(i) + ".jpg", frame);
			}
			cout << "cap image success!" << endl;
			break;
		}
		else {
			for (int i = 0; i < videos.size(); i++)
			{
				*videos[i] >> frame;
				putText(frame, get_current_time(), Point(0, 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 200, 200),2);
				MyUtil::showImg("camera" + to_string(i), frame);
			}
		}
		if (waitKey(33) == 27) break;
	}

	for (int i = 0; i < videos.size(); i++) {
		delete videos[i];
	}
}

string Camera::get_current_time()
{
#pragma warning(disable:4996) 
	auto now = chrono::system_clock::now();
	time_t tt = chrono::system_clock::to_time_t(now);
	tm* timeinfo = localtime(&tt);
	char buf[24];
	strftime(buf, 24, "%Y-%m-%d %H:%M:%S,", timeinfo);

	chrono::milliseconds ms = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch());

	string str(buf);
	string ms_str = to_string(ms.count() % 1000);
	if (ms_str.length() < 3) { 
		ms_str = string(3 - ms_str.length(), '0') + ms_str;
	}
	str += ms_str;
	return str;
}

void Camera::view_frame_detail(string path)
{
	VideoCapture cap(path);
	int i = 0;
	for (;;) {
		Mat frame;
		bool flag = cap.read(frame);
		if (flag) {
			cout << "read " << i + 1 << " frame" << endl;
			MyUtil::showImg("pano", frame);
			i++;
		}
		else {
			break;
		}
		if (waitKey(33) == 27) break;
	}

	cap.release();
}

void Camera::output_camera_args(VideoCapture* cap)
{
	double brightness = cap->get(CAP_PROP_BRIGHTNESS); 
	double contrast = cap->get(CAP_PROP_CONTRAST); 
	double saturatoin = cap->get(CAP_PROP_SATURATION); 
	double hue = cap->get(CAP_PROP_HUE); 
	double gain = cap->get(CAP_PROP_GAIN); 
	double exposure = cap->get(CAP_PROP_EXPOSURE); 
	double b_white_balance = cap->get(CAP_PROP_WHITE_BALANCE_BLUE_U); 
	double r_white_balance = cap->get(CAP_PROP_WHITE_BALANCE_RED_V); 

	cout << "brightness: " << brightness << ", contrast:" << contrast << ", saturation: " << saturatoin << ", hue: " << hue << ", gain: " << gain
		<< ", exposure: " << exposure << ", b_white_balance: " << b_white_balance << ", r_white_balance: " << r_white_balance << endl;
}
