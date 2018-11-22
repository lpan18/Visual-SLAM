#define EIGEN_DONT_ALIGN_STATICALLY
#ifndef EXTRACTOR_H_
#define EXTRACTOR_H_


#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

#include "frontend/VisualFrontend.h"
#include "backend/Backend.h"
#include <pangolin/pangolin.h>


class VisualOdometryLogic
{
public:
	VisualOdometryLogic(VisualFrontend& frontend, Backend& backend, double fx, double fy, double cx, double cy);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);
	virtual void handleImage(cv::Mat& image);
	Eigen::Affine3d getcurrentpose(){
		return T_WC;
	}

	virtual ~VisualOdometryLogic();

protected:
	//void publishFeatures(const std::string& frame_id, ros::Time stamp);
	void trackPose(Features2D& trackedFeatures, Features2D& newFeatures);
	void display(Mat& currentimage);


private:
	void drawFeatures(cv::Mat& frame, Features2D& features,
				cv::Scalar colorMatched, cv::Scalar colorNew);

private:
	//Visual Frontend
	VisualFrontend& frontend;

	//Localization Backend
	Backend& backend;

	//config
	//ConfigManager config;

	//Pose
	Eigen::Affine3d T_CR;
	Eigen::Affine3d T_RC;
	Eigen::Affine3d T_WC;
	Eigen::Affine3d Tgt;
	Matx33d K; // camera intrinsic


};

#endif /* EXTRACTOR_H_ */
