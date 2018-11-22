#include "logic/VisualOdometryLogic.h"
#include <opencv2/opencv.hpp>

#include "backend/Backend.h"
#include "backend/BackendSFM.h"

using namespace cv;
using namespace std;

cv::Mat Affinetocvmat(const Eigen::Affine3d T){
    cv::Mat cvMat(4,4,CV_64F);
    Eigen::Matrix4d m = T.matrix();
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<double>(i,j)=m(i,j);
	return  cvMat;
}

cv::Mat M44tocvmat(const Eigen::Affine3d T){
	cv::Mat cvMat(4,4,CV_64F);
	Eigen::Matrix4d m = T.matrix();
	for(int i=0;i<4;i++)
		for(int j=0; j<4; j++)
			cvMat.at<double>(i,j)=m(i,j);
	return  cvMat;
}

VisualOdometryLogic::VisualOdometryLogic(VisualFrontend& frontend,
			Backend& backend, double fx, double fy, double cx, double cy) :
			frontend(frontend), backend(backend)
{


	//Init rotation camera/robot
	Eigen::Quaterniond q_CR(0.5, 0.5, -0.5, 0.5);
	Eigen::Quaterniond q_RC(-0.5, 0.5, -0.5, 0.5);

	T_CR.setIdentity();
	T_CR.rotate(q_CR);

	T_RC.setIdentity();
	T_RC.rotate(q_RC);

	T_WC.setIdentity();

	//Init pose
	//T_WC = config.T_WR * T_RC;
	//Tgt = config.T_WR;

	K = cv::Mat_<double> ( 3,3 );

	K << fx, 0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;



	//Init Backend
	backend.setCameraPose(T_WC);
}


void VisualOdometryLogic::handleImage(Mat & currentimage)
{
		Mat image = currentimage.clone();

		Features2D trackedFeatures;
		Features2D newFeatures;

        auto festart = chrono::steady_clock::now();
		frontend.trackAndExtract(image, trackedFeatures, newFeatures);
        auto feend = chrono::steady_clock::now();
        cout << "feature extraction running time: "<< chrono::duration <double, milli> (feend-festart).count() << " ms" << endl;

		//Track pose
        auto trstart = chrono::steady_clock::now();
		trackPose(trackedFeatures, newFeatures);
        auto trend = chrono::steady_clock::now();
        cout << "tracking running time: "<< chrono::duration <double, milli> (trend-trstart).count() << " ms" << endl;

		//display image
		display(currentimage);
}

VisualOdometryLogic::~VisualOdometryLogic()
{

}

void VisualOdometryLogic::display(Mat& currentimage)
{
	//Print debug image
	Mat img = currentimage.clone();
	drawFeatures(img, frontend.getCurrentFeatures(), Scalar(255, 0, 0),
				Scalar(0, 255, 0));
	imshow("current img", img);
	waitKey(1);
}


void VisualOdometryLogic::trackPose(
			Features2D& trackedFeatures, Features2D& newFeatures)
{
	//Compute Camera Pose
	backend.setK(K);
	T_WC = backend.computePose(trackedFeatures, newFeatures);

	//Eigen::Affine3d T_WR = T_WC * T_CR;

}

void VisualOdometryLogic::drawFeatures(Mat& frame, Features2D& features,
			cv::Scalar colorMatched, cv::Scalar colorNew)
{

	static unsigned int max_id = 0;
	unsigned int new_max_id = 0;

	for (size_t j = 0; j < features.size(); j++)
	{
		cv::Scalar drawColor =
					(features.getId(j) > max_id) ? colorNew : colorMatched;
		circle(frame, features[j], 3, drawColor);

		new_max_id = std::max(new_max_id, features.getId(j));
	}

	max_id = new_max_id;
}

void VisualOdometryLogic::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
		cv::Mat Rwc(3,3,CV_64F);
		cv::Mat twc(3,1,CV_64F);
		{
			cv::Mat curr_Pose =Affinetocvmat(getcurrentpose()).clone();
			//cout<<"the pose we have\n"<<curr_Pose<<endl;
			Rwc = curr_Pose.rowRange(0,3).colRange(0,3);
			//cout<<"translation:\n"<<curr_Pose.rowRange(0,3).col(3)<<endl;
			twc = curr_Pose.rowRange(0,3).col(3);
		}

		M.m[0] = Rwc.at<double>(0,0);
		M.m[1] = Rwc.at<double>(1,0);
		M.m[2] = Rwc.at<double>(2,0);
		M.m[3]  = 0.0;

		M.m[4] = Rwc.at<double>(0,1);
		M.m[5] = Rwc.at<double>(1,1);
		M.m[6] = Rwc.at<double>(2,1);
		M.m[7]  = 0.0;

		M.m[8] = Rwc.at<double>(0,2);
		M.m[9] = Rwc.at<double>(1,2);
		M.m[10] = Rwc.at<double>(2,2);
		M.m[11]  = 0.0;

		M.m[12] = twc.at<double>(0);
		M.m[13] = twc.at<double>(1);
		M.m[14] = twc.at<double>(2);
		M.m[15]  = 1.0;
}
