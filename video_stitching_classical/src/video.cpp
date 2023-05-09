#include"video.h"
#include"my_util.h"

#include"image_stitch.h"
#include"warper.h"
#include"feature_finder.h"

vector<Mat> Video::cut_frame(Mat frame, Size grid, vector<int> order)
{
	if (grid.width * grid.height != order.size()) {
		cout << "the size of grid and order must be same.";
		return vector<Mat>(order.size());
	}
	int cell_width = ceil(float(frame.cols) / grid.width), cell_height = ceil(float(frame.rows) / grid.height);
	vector<Mat> frames;
	for (int i = 0; i < frame.rows; i+=cell_height) {
		for (int j = 0; j < frame.cols; j += cell_width) {
			int cw = min(cell_width, frame.cols - j), ch = min(cell_height, frame.rows - i);
			Mat cut = frame(Rect(j, i, cw, ch));
			frames.push_back(cut);
		}
	}
	
	vector<Mat> order_frames(frames.size());
	for (int i = 0; i < frames.size(); i++) {
		order_frames[i] = frames[order[i]];
	}
	return order_frames;
}

void Video::test_video()
{
	VideoCapture cap;
	string path = "../video_stitching_classical/videos/outside2.mp4";
	if (!cap.open(path)) {
		cout << "open video fail!" << endl;
		return;
	}

	int F = 0;
	vector<Mat> Hs;
	Mat pano;
	VideoWriter vw;

	ImageStitch stitch;
	Warper warper;
	FeatureFinder finder;

	for (;;) {
		if (cap.grab()) {
			Mat frame;
			bool is_null = cap.retrieve(frame);
			if (!is_null) break;
			cout << "frame " << ++F << " grap success!" << endl;

			vector<Mat> order_frames = cut_frame(frame, Size(2, 2), vector<int>{0, 1, 2, 3});
			vector<Mat> frames;
			for (int i = 0; i < order_frames.size(); i++) {
				if (i != 0) {
					frames.push_back(order_frames[i]);
				}
			}
			

			for (int k = 0; k < frames.size(); k++) {
				MyUtil::showImg("camera" + to_string(k), frames[k]);
			}

			if (F == 10) {
				for (int i = 1; i < frames.size(); i++) {
					Mat frame = warper.cylindrical_projection(frames[i]);
					Mat target = warper.cylindrical_projection(frames[i - 1]);
					Mat H = finder.find_H(frame, target);
					cout << "H" << i - 1 << "-" << i << ":" << H << endl;
					Hs.push_back(H);
				}
				stitch.mid_stitch(frames, Hs, pano);
				vw.open("pano.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0,
					Size(pano.cols, pano.rows), true);
				vw << pano;
			}
			if (F >= 10) {
				stitch.mid_stitch(frames, Hs, pano);
				vw << pano;
				MyUtil::showImg("pano", pano);
			}

			if (waitKey(33) == 27) break;
		}
	}
	cap.release();
}
