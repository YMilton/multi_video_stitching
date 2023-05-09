#include"video_stitch.h"
#include"stitch_detail.h"

using namespace stitch;

void image_stitch() {
	Mat img1 = imread("../multi_video_stitching/images/005.jpg");
	Mat img2 = imread("../multi_video_stitching/images/006.jpg");

	vector<Mat> imgs;
	imgs.push_back(img1);
	imgs.push_back(img2);

	ImageStitch is;
	is.imgs = imgs;
	is.stitch();

	namedWindow("pano", WINDOW_NORMAL);
	imshow("pano", is.pano);
	waitKey();
}


int main() {

	VideoStitch vs;
	vs.videoStitch();


	return 0;

}