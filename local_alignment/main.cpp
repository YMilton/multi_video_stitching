#include<opencv.hpp>
#include<opencv2\ximgproc\slic.hpp>

#include"layer_registration.h"
#include"seagull.h"

#include"image_stitch.h"
#include"video.h"
#include"camera.h"

using namespace cv;

/* video_stitching_classicalœÓƒø≤‚ ‘ */
void video_stitching_classical_test() {
	ImageStitch is;
	//is.test_videos_stitch();

	is.image_stitch_test();

	/*Video v;
	v.test_video();*/
}


int main(int argc, char** argv)
{
	video_stitching_classical_test();

	LayerRegistration lr;
	/*lr.test_fun();
	lr.test_layerH();*/

	SEAGULL seagull;
	//seagull.test();

	/*Camera c;
	c.save1Frame(vector<string>{"1"}, "./");*/

	return 0;
}