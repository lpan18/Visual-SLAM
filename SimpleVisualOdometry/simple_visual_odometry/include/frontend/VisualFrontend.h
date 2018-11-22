#define EIGEN_DONT_ALIGN_STATICALLY
#ifndef INCLUDE_VISUALFRONTEND_H_
#define INCLUDE_VISUALFRONTEND_H_

#include "core/Features.h"
#include "OccupancyGrid.h"
using namespace std;
using namespace cv;

class VisualFrontend
{
public:
	VisualFrontend();

	void trackAndExtract(cv::Mat& im_gray, Features2D& trackedPoints, Features2D& newPoints);

	inline Features2D& getCurrentFeatures()
	{
		return oldPoints;
	}

	// utils
	static void downloadmask(const GpuMat& d_mat, vector<uchar>& vec);
	static void downloadpts(const GpuMat& d_mat, vector<Point2f>& vec);


protected:
	void extract(cv::Mat& im_gray, Features2D& newPoints);
	void track(cv::Mat& im_gray, Features2D& points);

	// Implement two functions below
	void extract1(Mat& im_gray, Features2D& newPoints);
	void track1(cv::Mat& im_gray, Features2D& points);

protected:
	//extracted data
	Features2D oldPoints;
	cv::Mat im_prev;

private:
	//unique ID for each point
	unsigned int newId;

    //cv::Ptr<cv::FeatureDetector> detector;

	//klt tracker and feature detector
    GoodFeaturesToTrackDetector_GPU gpu_detector;
    PyrLKOpticalFlow d_pyrLK;
	//Data needed to guide extraction
	OccupancyGrid grid;

	//parameters
	//const int thresholdExtraction = 20;
	const double thresholdFBError = 1.0;

};



#endif /* INCLUDE_VISUALFRONTEND_H_ */
