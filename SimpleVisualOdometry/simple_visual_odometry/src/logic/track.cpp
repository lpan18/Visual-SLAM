//
// Created by sicong on 31/10/18.
//

#include "frontend/VisualFrontend.h"
#include <chrono>
#include "frontend/OccupancyGrid.h"
#include "backend/BackendSFM.h"
#include "backend/g2o_types.h"

using namespace std;
using namespace cv;


void VisualFrontend::extract(Mat& im_gray, Features2D& newPoints)
{
	vector<Point2f> newPointsVector;
	//detector->detect(im_gray, newPointsVector);


    //GoodFeaturesToTrackDetector_GPU detector(300, 0.01, 0);
    GpuMat d_frame0Gray(im_gray);
    GpuMat d_prevPts;

    gpu_detector(d_frame0Gray, d_prevPts);

    downloadpts(d_prevPts, newPointsVector);

	//Prepare grid
	grid.setImageSize(im_gray.cols, im_gray.rows);

#pragma omp parallel
	for (Point2f& oldPoint : oldPoints)
	{
		grid.addPoint(oldPoint);
	}
	for (auto point : newPointsVector)
	{
		if (grid.isNewFeature(point))
		{
			oldPoints.addPoint(point, newId);
			newPoints.addPoint(point, newId);
			newId++;
		}
	}
	grid.resetGrid();
}


void VisualFrontend::track(Mat& im_gray, Features2D& trackedPoints)
{
    //vector<unsigned char> status;
    //vector<unsigned char> status_back;
    //vector<Point2f> pts_back;
    //vector<Point2f> nextPts;
    //	vector<float> err;
    //	vector<float> err_back;

    vector<Point2f> points = oldPoints.getPoints();
    vector<float> fb_err;
    Mat prevPts = Mat(1,points.size(), CV_32FC2);
    //Calculate forward optical flow for prev_location


    for(size_t i=0;i<points.size();i++){
        prevPts.at<Vec2f>( 0, i )[0]=points[i].x;
        prevPts.at<Vec2f>( 0, i )[1]=points[i].y;
    }



    GpuMat d_frame0(im_prev);
    GpuMat d_frame1(im_gray);
    GpuMat d_nextPts;
    GpuMat d_status,d_status_back;
    GpuMat d_pts_back;


    GpuMat d_prevPts(prevPts);

    d_pyrLK.sparse(d_frame0, d_frame1, d_prevPts, d_nextPts, d_status);

    vector<Point2f> nextPts(d_nextPts.cols);
    downloadpts(d_nextPts, nextPts);

    vector<uchar> status(d_status.cols);
    downloadmask(d_status, status);

    d_pyrLK.sparse(d_frame1, d_frame0, d_nextPts, d_pts_back, d_status_back);

    vector<Point2f> pts_back(d_pts_back.cols);
    downloadpts(d_pts_back, pts_back);

    vector<uchar> status_back(d_status_back.cols);
    downloadmask(d_status_back, status_back);


    //cpu version

//	calcOpticalFlowPyrLK(im_prev, im_gray, points, nextPts, status, err);
//	//Calculate backward optical flow for prev_location
//	calcOpticalFlowPyrLK(im_gray, im_prev, nextPts, pts_back, status_back,
//				err_back);



    //Calculate forward-backward error
    for (size_t i = 0; i < points.size(); i++)
    {
        fb_err.push_back(norm(pts_back[i] - points[i]));
    }

    //Set status depending on fb_err and lk error
#pragma omp parallel
    for (size_t i = 0; i < status.size(); i++)
        status[i] = (fb_err[i] <= thresholdFBError) && status[i];

    trackedPoints = Features2D(oldPoints, nextPts, status);

}




void OccupancyGrid::initializer()
{
    Iy = 1;
    Ix = 1;

    resetGrid();
}

void OccupancyGrid::setImageSize(size_t cols, size_t rows)
{
    Ix = cols / nx;
    Iy = rows / ny;
}

void OccupancyGrid::addPoint(Point2f& p)
{
    size_t i = p.x / Ix;
    size_t j = p.y / Iy;

    if(i >= nx || j >= ny)
        return;

    isFree[i][j] = false;
}

bool OccupancyGrid::isNewFeature(Point2f& p)
{
    int i = p.x / Ix;
    int j = p.y / Iy;

    bool isNew = true;

    unsigned int minX = std::max(0, i - 1);
    unsigned int maxX = std::min((int)nx, i + 2);

    unsigned int minY = std::max(0, j - 1);
    unsigned int maxY = std::min((int)ny, j + 2);

    for(unsigned int x = minX; x < maxX; x++)
        for(unsigned int y = minY; y < maxY; y++)
            isNew = isNew && isFree[x][y];


    return isNew;
}

void OccupancyGrid::resetGrid()
{
    for (size_t i = 0; i < nx; i++)
        for (size_t j = 0; j < ny; j++)
            isFree[i][j] = true;
}

void BackendSFM::tracking(Features2D& trackedFeatures, Features2D& newFeatures)
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
	size_t num_inliers_ = features3D.getPoints().size();
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
    for (size_t i = 0; i < num_inliers_; i++) {
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
