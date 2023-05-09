#include "stitch_detail.h"

namespace stitch {
	ImageStitch::ImageStitch()
	{
		finder = xfeatures2d::SIFT::create(1200);
	}

	ImageStitch::ImageStitch(int detector)
	{
		if (detector == SIFT) {
			finder = xfeatures2d::SIFT::create(1000);
		}
		if (detector == SURF) {
			finder = xfeatures2d::SURF::create(1000);
		}
		if (detector == ORB) {
			finder = ORB::create(1000);
		}
	}

	void ImageStitch::getCameraArgs()
	{
		double t = getTickCount();
		vector<ImageFeatures> features;
		computeImageFeatures(finder, imgs, features);

		vector<MatchesInfo> pairwise_matches;
		BestOf2NearestMatcher matcher;
		matcher(features, pairwise_matches);

		HomographyBasedEstimator estimator;
		estimator(features, pairwise_matches, cameras);
		for (size_t i = 0; i < cameras.size(); i++) { 
			Mat R;
			cameras[i].R.convertTo(R, CV_32F);
			cameras[i].R = R;
		}
		Ptr<BundleAdjusterBase> adjuster;
		adjuster = new BundleAdjusterRay();  
		adjuster->setConfThresh(1); 
		(*adjuster)(features, pairwise_matches, cameras); 
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); i++)
			rmats.push_back(cameras[i].R);
		waveCorrect(rmats, WAVE_CORRECT_HORIZ); 
		for (size_t i = 0; i < cameras.size(); i++)
			cameras[i].R = rmats[i];

		cout << "get camera args time: " << (getTickCount() - t) / getTickFrequency() << endl;
	}

	void ImageStitch::warpImages()
	{
		double t = getTickCount();

		vector<Mat> srcMasks; 
		warpImgs.resize(imgs.size());
		warpMasks.resize(imgs.size());
		topCorners.resize(imgs.size());
		for (size_t i = 0; i < imgs.size(); i++) { 
			Mat src(imgs[i].size(), CV_8U);
			src.setTo(Scalar::all(255)); 
			srcMasks.push_back(src);
		}
		float mid_focal;
		int mid_idx = cameras.size() / 2;
		if (cameras.size() % 2 == 1) {
			mid_focal = cameras[mid_idx].focal;
		}
		else {
			mid_focal = (cameras[mid_idx - 1].focal + cameras[mid_idx].focal) / 2;
		}

		Ptr<WarperCreator> warp_creator;
		warp_creator = new cv::CylindricalWarper(); 

		Ptr<RotationWarper> warper = warp_creator->create(mid_focal);
		for (size_t i = 0; i < cameras.size(); i++) {
			Mat K;
			cameras[i].K().convertTo(K, CV_32F);  
			topCorners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, warpImgs[i]);
			warper->warp(srcMasks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, warpMasks[i]);
		}

		cout << "warp image time: " << (getTickCount() - t) / getTickFrequency() << endl;
	}

	void ImageStitch::findSeam()
	{
		double t = getTickCount();

		Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN); 
		vector<UMat> warp_imgs(imgs.size()), warp_masks(imgs.size());
		for (size_t i = 0; i < imgs.size(); i++) {
			warpImgs[i].copyTo(warp_imgs[i]);
			warpMasks[i].copyTo(warp_masks[i]);
		}
		compensator->feed(topCorners, warp_imgs, warp_masks);
		for (size_t i = 0; i < imgs.size(); i++) {
			compensator->apply(i, topCorners[i], warp_imgs[i], warp_masks[i]);
		}

		Ptr<SeamFinder> seam_finder;
		seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD); 
		for (size_t i = 0; i < warp_imgs.size(); i++) {
			warp_imgs[i].convertTo(warp_imgs[i], CV_32F);
		}
		seam_finder->find(warp_imgs, topCorners, warp_masks);
		for (size_t i = 0; i < imgs.size(); i++) {
			warp_imgs[i].copyTo(warpImgs[i]);
			warp_masks[i].copyTo(warpMasks[i]);
		}

		cout << "find seam and image exposure time: " << (getTickCount() - t) / getTickFrequency() << endl;
	}


	void ImageStitch::blendImages()
	{
		double t = getTickCount();

		Ptr<Blender> blender; 
		blender = new FeatherBlender(0.8); 
		vector<Size> sizes;
		for (size_t i = 0; i < warpImgs.size(); i++) {
			sizes.push_back(warpImgs[i].size());
		}
		blender->prepare(topCorners, sizes); 
		vector<Mat> warp_imgs_s(topCorners.size());
		for (size_t i = 0; i < warpImgs.size(); i++) {
			warpImgs[i].convertTo(warp_imgs_s[i], CV_16S); 
			blender->feed(warp_imgs_s[i], warpMasks[i], topCorners[i]);
		}
		Mat pano_mask;
		blender->blend(pano, pano_mask);
		pano.convertTo(pano, CV_8UC3);

		cout << "blend image time: " << (getTickCount() - t) / getTickFrequency() << endl;
	}


	Mat ImageStitch::stitch()
	{
		if (imgs.size() == 0) {
			cout << "do not input image." << endl;
			return Mat();
		}

		getCameraArgs();

		warpImages();

		findSeam();

		blendImages();

		return pano;
	}
}
