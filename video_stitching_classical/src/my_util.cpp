#include<io.h>
#include<direct.h>

#include"my_util.h"


void MyUtil::showImg(string winName, Mat img)
{
	namedWindow(winName, WINDOW_NORMAL);
	imshow(winName, img);
}

void MyUtil::createDir(string path)
{
	if (_access(path.c_str(), 0) == -1) {
		int code = _mkdir(path.c_str());
	}
}


void MyUtil::write_mat(string fileName, Mat mat)
{
	mat.convertTo(mat, CV_32F);
	ofstream f(fileName,ios::binary);
	if (!f.is_open())
	{
		cout << "cannot open file." << endl;
		return;
	}
	
	for (int i = 0; i < mat.rows; i++)
	{
		float* p = mat.ptr<float>(i);
		for (int j = 0; j < mat.cols; j++)
		{
			f << int(p[j]) << " ";
		}
		f << endl;
	}

}

float MyUtil::euclid(Point2f p1, Point2f p2)
{
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)); 
}

float MyUtil::L2_square(Point2f p1, Point2f p2)
{
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}
