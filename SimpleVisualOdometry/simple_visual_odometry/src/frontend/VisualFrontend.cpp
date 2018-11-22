#include "frontend/VisualFrontend.h"
#include <chrono>
using namespace std;
using namespace cv;

void VisualFrontend::downloadpts(const GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

void VisualFrontend::downloadmask(const GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

VisualFrontend::VisualFrontend()
{
	//Initialise detector
//	std::string detectorType = "Feature2D.BRISK";
//
//	detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", thresholdExtraction);
	//Initialize ID
    gpu_detector = GoodFeaturesToTrackDetector_GPU(250, 0.01, 0);

    d_pyrLK.winSize.width = 21;
    d_pyrLK.winSize.height = 21;
    d_pyrLK.maxLevel = 3;
    d_pyrLK.iters = 30;

	newId = 0;

}

void VisualFrontend::trackAndExtract(cv::Mat& im_gray, Features2D& trackedPoints, Features2D& newPoints)
{
	if (oldPoints.size() > 0)
	{
        //Track prevoius points with optical flow
		auto festart = chrono::steady_clock::now();
		track1(im_gray, trackedPoints);
		auto feend = chrono::steady_clock::now();
		cout << "klt running time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;


		//Save tracked points
		oldPoints = trackedPoints;
	}

	//Extract new points
	auto festart = chrono::steady_clock::now();
	extract1(im_gray, newPoints);
	auto feend = chrono::steady_clock::now();
	cout << "new feature time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;

	//save old image
	im_prev = im_gray;

}

void VisualFrontend::extract1(Mat& im_gray, Features2D& newPoints)
{
    // TODO
	// reset grid with old feature points
	vector<Point2f> oldFeaturePoints = oldPoints.getPoints();
	grid.setImageSize1(im_gray.cols, im_gray.rows);
	for (size_t i = 0; i < oldFeaturePoints.size(); i++) {
		grid.addPoint1(oldFeaturePoints[i]);
	}

	// Pump up to GPU
	GpuMat d_frame_0(im_gray);
	GpuMat d_curr_pts;
	gpu_detector(d_frame_0, d_curr_pts);

	// Save detected points
	vector<Point2f> kps(d_curr_pts.cols);
	downloadpts(d_curr_pts, kps);

	// if new points, add
	for (size_t i = 0; i < kps.size(); i++) {
		Point2f curr_pt = kps[i];
		if (grid.isNewFeature1(curr_pt)) {
			oldPoints.addPoint(curr_pt, newId);
			newPoints.addPoint(curr_pt, newId);
			newId++;
		}
	}

	// reset grid
	grid.resetGrid1();
}

void VisualFrontend::track1(Mat& im_gray, Features2D& trackedPoints)
{
    // TODO

	// init vectors
	vector<cv::Point2f> next_keypoints;
	vector<cv::Point2f> prev_keypoints;
	vector<cv::Point2f> back_keypoints;
	vector<unsigned char> forward_status;
    vector<unsigned char> backward_status;

	// old features to cv mat to gpu mat
	prev_keypoints = oldPoints.getPoints();
	Mat prev_keypoints_mat = Mat(1, prev_keypoints.size(), CV_32FC2, (void*)&prev_keypoints[0]);

	// init gpu variables
	GpuMat d_frame0(im_prev);
	GpuMat d_frame1(im_gray);
	GpuMat d_prev_pts(prev_keypoints_mat);
	GpuMat d_next_pts;
	GpuMat d_back_pts;
	GpuMat d_status;
	GpuMat d_back_status;

	// forward
	d_pyrLK.sparse(d_frame0, d_frame1, d_prev_pts, d_next_pts, d_status);

	next_keypoints.resize(d_next_pts.cols);
	forward_status.resize(d_status.cols);
	downloadpts(d_next_pts, next_keypoints);
	downloadmask(d_status, forward_status);

	// backward
	d_pyrLK.sparse(d_frame1, d_frame0, d_next_pts, d_back_pts, d_back_status);

	back_keypoints.resize(d_back_pts.cols);
	backward_status.resize(d_back_status.cols);
	downloadpts(d_back_pts, back_keypoints);
	downloadmask(d_back_status, backward_status);

	// Compare distance between prev and back
	vector<unsigned char> status(forward_status.size());
	
	for (size_t idx = 0; idx < back_keypoints.size(); idx++) {
        double pt_dist = norm(back_keypoints[idx] - prev_keypoints[idx]);
        status[idx] = (pt_dist < thresholdFBError) && forward_status[idx] && backward_status[idx];
    }

	trackedPoints = Features2D(oldPoints, next_keypoints, status);
}
