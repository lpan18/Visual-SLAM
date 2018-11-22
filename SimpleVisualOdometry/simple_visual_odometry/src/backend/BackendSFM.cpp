#include "backend/BackendSFM.h"
#include "backend/g2o_types.h"
#include "core/utils.h"

BackendSFM::BackendSFM()
{
	Fpoints.setIdentity();
}

Features3D BackendSFM::getFeatures() const
{
	return new3DPoints;
}

void BackendSFM::prelude()
{
	//Clear new points
	new3DPoints = Features3D();
}

void BackendSFM::startup(Features2D& trackedFeatures, Features2D& newFeatures)
{
	if (newFeatures.size() + trackedFeatures.size() > minInitialFeatures)
	{
		//Accept first features
		oldFeatures = trackedFeatures;
		oldFeatures.addPoints(newFeatures);
		//Update state
		state = Initializing;
		std::cout << "Initializing" << std::endl;
	}
}

void BackendSFM::initialization(Features2D& trackedFeatures,
			Features2D& newFeatures)
{
	if (trackedFeatures.size() < minFeatures)
		throw Backend::no_points_exception();

	Features2D oldCorrespondences;
	Features2D newCorrespondences;

	double deltaMean = getCorrespondecesAndDelta(oldFeatures, trackedFeatures,
				oldCorrespondences, newCorrespondences);

	if (oldCorrespondences.size() < minFeatures)
		throw Backend::no_points_exception();

	if (sufficientDelta(deltaMean))
	{
		//compute transform
		vector<unsigned char> mask;
		Mat E;
		Mat C;
		recoverCameraFromEssential(oldCorrespondences, newCorrespondences, mask,
					C, E);

		//compute absolute cameras
		cv::Mat C0, C1;
		computeInitialCameras(C, C0, C1);

		//triangulate
		Features3D&& triangulated = triangulate(oldCorrespondences,
					newCorrespondences, C1, C0);

		//Save points
		old3DPoints = triangulated;
		new3DPoints = triangulated;

		//Compute new pose
		T_WC = cameraToTransform(C1);

		//Save new features in candidate list
		if (newFeatures.size() > 0)
		{
			KeyFrame candidate(C, T_WC, newFeatures);
			keyframes.push_front(candidate);
		}

		//Update state
		state = Tracking;
		std::cout << "Tracking" << std::endl;
	}
}

void BackendSFM::tracking1(Features2D& trackedFeatures, Features2D& newFeatures)
{

	//Compute motion
	Features2D features2D;
	Features3D features3D;

	getCorrespondences(trackedFeatures, features2D, features3D);

	Eigen::Affine3d T_CW = T_WC.inverse();
	cv::Mat rvec = rodriguesFromPose(T_CW);
	cv::Mat tvec = translationFromPose(T_CW);

	solvePnP(features3D.getPoints(), features2D.getPoints(), K, Mat(), rvec,
			 tvec, true);
	int num_inliers_ = features3D.getPoints().size();
//
//	Mat inliers;
//	solvePnPRansac( features3D.getPoints(), features2D.getPoints(), K, Mat(), rvec, tvec, false, 100, 8.0, 100, inliers, 1 );
//    int num_inliers_ = inliers.rows;
//    std::cout<<"pnp inliers: "<<num_inliers_<< std::endl;


    SE3 T_c_w_estimated_ = SE3(
            SO3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)),
            Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))
    );
    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block *solver_ptr = new Block(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    //std::cout<<"Initial pose: "<<T_c_w_estimated_.rotation_matrix()<<"\nInitial pose:\n"<< T_c_w_estimated_.translation()<<std::endl;
    pose->setEstimate(g2o::SE3Quat(
            T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
    ));
    optimizer.addVertex(pose);
    // edges
    for (int i = 0; i < num_inliers_; i++) {
        int index = i; //inliers.at<int>(i, 0);
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly *edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId(i);
        edge->setVertex(0, pose);
        edge->fx_ = (double) K(0, 0);
        edge->fy_ = (double) K(1, 1);
        edge->cx_ = (double) K(0, 2);
        edge->cy_ = (double) K(1, 2);
        edge->point_ = Vector3d(features3D.getPoints()[index].x, features3D.getPoints()[index].y,
                                features3D.getPoints()[index].z);
        edge->setMeasurement(Vector2d(features2D.getPoints()[index].x, features2D.getPoints()[index].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }

    std::cout << "graph done!" << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    T_c_w_estimated_ = SE3(
            pose->estimate().rotation(),
            pose->estimate().translation()
    );

    std::cout << "T_c_w_estimated_: " << std::endl << T_c_w_estimated_.matrix() << std::endl;


    //cv::Mat C = computeCameraMatrix(rvec, tvec);

    cv::Mat C = computeCameraMatrixeigen(T_c_w_estimated_.matrix());


        //Compute new pose
        //T_WC = cameraToTransform(C);
    T_WC = M44tAffine(T_c_w_estimated_.matrix());

	//Compute structure
	computeStructure(C, trackedFeatures);

	//Generate keyframes
	generateKeyframe(C, trackedFeatures, newFeatures);
}

void BackendSFM::recovery(Features2D& trackedFeatures, Features2D& newFeatures)
{
	startup(trackedFeatures, newFeatures);
}

void BackendSFM::generateKeyframe(const cv::Mat& C, Features2D& trackedFeatures,
			Features2D& newFeatures)
{
	const unsigned int minpoints = 20;

	if (toTriangulate.size() > minpoints)
	{
		Features2D features;

		//find features to triangulate
		for(auto id : toTriangulate)
		{
			if(trackedFeatures.contains(id))
			{
				unsigned int index = trackedFeatures.getIndex(id);

				features.addPoint(trackedFeatures[index], id);
			}

		}

		//Add new keyframe
		if(trackedFeatures.size() > minpoints/2)
		{
			KeyFrame keyframe(C, T_WC, features);
			keyframes.push_back(keyframe);
		}

		//Clear list
		toTriangulate.clear();
	}

	// Add new features to next keyframe
	for (unsigned int i = 0; i < newFeatures.size(); i++)
		toTriangulate.insert(newFeatures.getId(i));

}

void BackendSFM::generateKeyframe2(const cv::Mat& C,
			Features2D& trackedFeatures, Features2D& newFeatures)
{
	//Save new features in candidate list
	if (newFeatures.size() > 0)
	{
		KeyFrame candidate(C, T_WC, newFeatures);
		keyframes.push_back(candidate);
	}
}

void BackendSFM::computeStructure(const cv::Mat& C, Features2D& trackedFeatures)
{
	//Compute structure
	auto it = keyframes.begin();
	while (it != keyframes.end())
	{
		//get candidate
		auto& keyframe = *it;

		//get translation between frames
		Eigen::Vector3d tf = keyframe.F.translation();
		Eigen::Vector3d tc = T_WC.translation();

		if ((tf - tc).norm() > 1.0)
		{
			// Find correspondences
			Features2D oldCorrespondences;
			Features2D newCorrespondences;
			getCorrespondecesAndDelta(keyframe.features, trackedFeatures,
						oldCorrespondences, newCorrespondences);
			//Triangulate correspondent features
			if (oldCorrespondences.size() > 0)
			{
				Mat E = computeEssential(keyframe.F, T_WC);
				Mat K = Mat(this->K);
				Mat F = K.t().inv() * E * K.inv();
				vector<Point2f> point1, point2;

				correctMatches(F, oldCorrespondences.getPoints(),
							newCorrespondences.getPoints(), point1, point2);

				oldCorrespondences = Features2D(oldCorrespondences, point1);
				newCorrespondences = Features2D(newCorrespondences, point2);

				Features3D&& triangulated = triangulate(oldCorrespondences,
							newCorrespondences, C, keyframe.C);

				vector<unsigned char> cheiralityMask;
				cheiralityCheck(T_WC, keyframe.F, triangulated, cheiralityMask);

				std::cout << "inliers " << countInlier(cheiralityMask)
							<< std::endl;

				//Add 3d points to new points set
				new3DPoints.addPoints(triangulated, cheiralityMask);

				std::cout << "Triangulated " << triangulated.size() << " points"
							<< std::endl;
			}

			//Erase processed candidate
			it = keyframes.erase(it);
		}
		else
		{
			//Update iterator
			it++;
		}
	}

	//Add triangulated points to points set
	old3DPoints.addPoints(new3DPoints);
}

void BackendSFM::cheiralityCheck(const Eigen::Affine3d& T1,
			const Eigen::Affine3d& T2, const Features3D& triangulated,
			std::vector<unsigned char>& mask)
{
	mask.resize(triangulated.size(), 1);
#pragma omp parallel
	for (unsigned int i = 0; i < triangulated.size(); i++)
	{
		auto pcv = triangulated[i];
		Eigen::Vector3d p(pcv.x, pcv.y, pcv.z);

		auto p1 = T1.inverse() * p;
		auto p2 = T2.inverse() * p;

		if (p1(2) < 0 || p2(2) < 0)
			mask[i] = 0;
	}
}

void BackendSFM::getCorrespondences(const Features2D& trackedFeatures,
			Features2D& features2D, Features3D& features3D)
{
    std::cout<<"find correspondences"<<std::endl;
    unsigned int ids[trackedFeatures.size()];
    bool ifin[trackedFeatures.size()];

#pragma omp parallel
	for (unsigned int i = 0; i < trackedFeatures.size(); i++)
	{
	    //std::cout<<i<<std::endl;
		unsigned int id = trackedFeatures.getId(i);
		ids[i] = id;
		if (old3DPoints.contains(id))
			ifin[i] = true;
		else
			ifin[i] = false;
	}

for(int i =0; i< trackedFeatures.size();i++)
{
	if(ifin[i])
	{
	unsigned int index = old3DPoints.getIndex(ids[i]);

	features2D.addPoint(trackedFeatures[i], ids[i]);
	features3D.addPoint(old3DPoints[index], ids[i]);
	}
}
	std::cout<<"done"<<std::endl;
}

//void BackendSFM::getCorrespondences1(const Features2D& trackedFeatures,
//									Features2D& features2D, Features3D& features3D)
//{
//	std::cout<<"find correspondences"<<std::endl;
//	for (unsigned int i = 0; i < trackedFeatures.size(); i++)
//	{
//		//std::cout<<i<<std::endl;
//		unsigned int id = trackedFeatures.getId(i);
//		if (old3DPoints.contains(id))
//		{
//			unsigned int index = old3DPoints.getIndex(id);
//			features2D.addPoint(trackedFeatures[i], id);
//			features3D.addPoint(old3DPoints[index], id);
//		}
//	}
//
//}

void BackendSFM::computeInitialCameras(cv::Mat C, cv::Mat& C0, cv::Mat& C1)
{
	Eigen::Affine3d T = cameraToTransform(C);

	Eigen::Affine3d T_WC1 = T_WC * T;

	C0 = transformToCamera(T_WC);
	C1 = transformToCamera(T_WC1);
}

cv::Mat BackendSFM::rodriguesFromPose(const Eigen::Affine3d& T)
{
	Eigen::Matrix3d R = T.rotation();
	Mat Rcv = (Mat_<double>(3, 3) <<  //
				R(0, 0), R(0, 1), R(0, 2), //
	R(1, 0), R(1, 1), R(1, 2), //
	R(2, 0), R(2, 1), R(2, 2));

	Mat rvec;
	Rodrigues(Rcv, rvec);

	return rvec;
}

cv::Mat BackendSFM::translationFromPose(const Eigen::Affine3d& T)
{
	Mat_<double> t(3, 1);

	t(0, 0) = T.translation().x();
	t(1, 0) = T.translation().y();
	t(2, 0) = T.translation().z();

	return t;
}

cv::Mat BackendSFM::computeCameraMatrix(const cv::Mat& rvec, const cv::Mat& t)
{
	Mat R;
	Rodrigues(rvec, R);

	Mat C;
	hconcat(R, t, C);

	return C;

}

cv::Mat BackendSFM::computeCameraMatrixeigen(const Eigen::Matrix4d& Tm)
{

	Mat C = (Mat_<double>(3, 4) <<  //
								Tm(0, 0), Tm(0, 1), Tm(0, 2), Tm(0, 3),//
			Tm(1, 0), Tm(1, 1), Tm(1, 2), Tm(1, 3),//
			Tm(2, 0), Tm(2, 1), Tm(2, 2), Tm(2, 3));

	return C;
}