////
//// Created by sicong on 08/11/18.
////
//
//#include <iostream>
//#include <fstream>
//#include <list>
//#include <vector>
//#include <chrono>
//using namespace std;
//
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/video/tracking.hpp>
//
//using namespace cv;
//int main( int argc, char** argv )
//{
//
//    if ( argc != 3 )
//    {
//        cout<<"usage: feature_extraction img1 img2"<<endl;
//        return 1;
//    }
//    //-- Read two images
//    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
//    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
//
//    list< cv::Point2f > keypoints;
//    vector<cv::KeyPoint> kps;
//
//    std::string detectorType = "Feature2D.BRISK";
//    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", 100);
//
//
//    detector->detect( img_1, kps );
//    for ( auto kp:kps )
//        keypoints.push_back( kp.pt );
//
//    vector<cv::Point2f> next_keypoints;
//    vector<cv::Point2f> prev_keypoints;
//    for ( auto kp:keypoints )
//        prev_keypoints.push_back(kp);
//    vector<unsigned char> status;
//    vector<float> error;
//    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
//    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
//    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
//
//    // visualize all  keypoints
//    hconcat(img_1,img_2,img_1);
//    for ( int i=0; i< prev_keypoints.size() ;i++)
//    {
//        cout<<(int)status[i]<<endl;
//        if(status[i] == 1)
//        {
//            Point pt;
//            pt.x =  next_keypoints[i].x + img_2.size[1];
//            pt.y =  next_keypoints[i].y;
//
//            line(img_1, prev_keypoints[i], pt, cv::Scalar(0,255,255));
//        }
//    }
//
//    cv::imshow("klt tracker", img_1);
//    cv::waitKey(0);
//
//    return 0;
//}

//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <math.h>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

bool checkinlier(cv::Point2f prev_keypoint, cv::Point2f next_keypoint, cv::Matx33d Fcandidate, double d)
{
    double u1 = prev_keypoint.x;
    double v1 = prev_keypoint.y;
    double u2 = next_keypoint.x;
    double v2 = next_keypoint.y;

    // epipolar line 1 to 2
    double a2 = Fcandidate(0, 0) * u1 + Fcandidate(0, 1) * v1 + Fcandidate(0, 2);
    double b2 = Fcandidate(1, 0) * u1 + Fcandidate(1, 1) * v1 + Fcandidate(1, 2);
    double c2 = Fcandidate(2, 0) * u1 + Fcandidate(2, 1) * v1 + Fcandidate(2, 2);

    double dist = (double)abs(a2 * u2 + b2 * v2 + c2) / sqrt(a2 * a2 + b2 * b2);

    // cout << "dist" << dist << endl;
    return dist <= d;
}

cv::Matx33d Findfundamental(vector<cv::Point2f> prev_subset, vector<cv::Point2f> next_subset, int img_w, int img_h, cv::Mat norm)
{
    // 8 * 9, the last row should be 0
    cv::Mat W = cv::Mat::zeros(prev_subset.size(), 9, CV_64F);
    for (size_t i = 0; i < prev_subset.size(); i++)
    {
        cv::Mat prev = (Mat_<double>(3, 1) << prev_subset[i].x, prev_subset[i].y, 1);
        cv::Mat next = (Mat_<double>(3, 1) << next_subset[i].x, next_subset[i].y, 1);

        cv::Mat prev_norm = norm * prev;
        cv::Mat next_norm = norm * next;

        double u1 = prev_norm.at<double>(0, 0);
        double v1 = prev_norm.at<double>(1, 0);
        double u2 = next_norm.at<double>(0, 0);
        double v2 = next_norm.at<double>(1, 0);
        double curr_point[] = {u1 * u2, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1.0};
        cv::Mat curr_row = cv::Mat(1, 9, CV_64F, curr_point);
        curr_row.copyTo(W.row(i));
    }

    // first SVD
    cv::SVD svd1(W); // get svd.vt, svd.w, svd.u;

    // second SVD
    cv::Mat e_hat = cv::Mat(3, 3, CV_64F);

    for (int i = 0; i < 9; i++)
    {
        e_hat.at<double>(i / 3, i % 3) = svd1.vt.at<double>((svd1.vt.rows - 1), i);
    }

    cv::SVD svd2(e_hat);
    cv::Mat w;
    svd2.w.copyTo(w);

    cv::Mat w_hat = cv::Mat::zeros(3, 3, CV_64F);
    w_hat.at<double>(0, 0) = w.at<double>(0, 0);
    w_hat.at<double>(1, 1) = w.at<double>(1, 0);

    cv::Mat F_hat = svd2.u * w_hat * svd2.vt;

    // denormalize points
    cv::Mat F_norm = norm.t() * F_hat * norm;
    F_norm = F_norm / F_norm.at<double>(2, 2);

    return (Matx33d)F_norm;
}

// Draw epipolar lines
static void drawEpipolarLines(const cv::Matx33d F,
                              const cv::Mat &img1, const cv::Mat &img2,
                              const std::vector<cv::Point2f> prev_subset,
                              const std::vector<cv::Point2f> next_subset,
                              const double d)
{
    cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
    cv::Rect rect1(0, 0, img1.cols, img1.rows);
    cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));

    cv::RNG rng(0);
    for (size_t i = 0; i < prev_subset.size(); i++)
    {
        cv::Scalar color(rng(256), rng(256), rng(256));

        double u1 = prev_subset[i].x;
        double v1 = prev_subset[i].y;
        double u2 = next_subset[i].x;
        double v2 = next_subset[i].y;

        cv::Matx33d F_t = F.t();

        // epipolar line 1
        double a1 = F_t(0, 0) * u2 + F_t(0, 1) * v2 + F_t(0, 2);
        double b1 = F_t(1, 0) * u2 + F_t(1, 1) * v2 + F_t(1, 2);
        double c1 = F_t(2, 0) * u2 + F_t(2, 1) * v2 + F_t(2, 2);

        // epipolar line 2
        double a2 = F(0, 0) * u1 + F(0, 1) * v1 + F(0, 2);
        double b2 = F(1, 0) * u1 + F(1, 1) * v1 + F(1, 2);
        double c2 = F(2, 0) * u1 + F(2, 1) * v1 + F(2, 2);

        double dist1 = (double)abs(a1 * u1 + b1 * v1 + c1) / sqrt(a1 * a1 + b1 * b1);
        double dist2 = (double)abs(a2 * u2 + b2 * v2 + c2) / sqrt(a2 * a2 + b2 * b2);

        if (dist1 > d || dist2 > d)
        {
            continue;
        }

        cv::Mat left = outImg(rect1);
        cv::Mat right = outImg(rect2);
        cv::line(left,
                 cv::Point(0, -c1 / b1),
                 cv::Point(img1.cols, -(c1 + a1 * img1.cols) / b1), color);
        cv::circle(left, prev_subset[i], 5, color, -1, CV_AA);

        cv::line(right,
                 cv::Point(0, -c2 / b2),
                 cv::Point(img2.cols, -(c2 + a2 * img2.cols) / b2), color);
        cv::circle(right, next_subset[i], 5, color, -1, CV_AA);
    }
    cv::imshow("Epipolar Constraint", outImg);
    cv::waitKey(0);
}

cv::Mat FindEssential(cv::Matx33d F, cv::Mat K)
{

    cv::Mat F_mat(F);
    cv::Mat E = K.t() * F_mat * K;
    return E;
}

void FindProjection(cv::Mat R, cv::Mat t, cv::Mat K_proj, cv::Mat &Rt, cv::Mat &P2)
{
    // R is 3x3, t is 3x1
    cv::Mat extra_zeros = cv::Mat::zeros(1, 3, CV_64F);
    cv::Mat R_cols;
    vconcat(R, extra_zeros, R_cols);

    cv::Mat extra_ones = cv::Mat::ones(1, 1, CV_64F);
    cv::Mat t_cols;
    vconcat(t.t(), extra_ones, t_cols);

    // P should be 4x4 here
    hconcat(R_cols, t_cols, Rt);

    // P should be 3x4 here
    P2 = K_proj * Rt;
    // cout << "P2 \n" << P2 << endl;
}

bool validateProjection(cv::Mat Rt, cv::Point2f pt1, cv::Point2f pt2, cv::Mat P1, cv::Mat P2)
{
    // Get A matrixes for both points, P1
    cv::Mat A1 = cv::Mat::zeros(2, 4, CV_64F);
    cv::Mat temp1 = pt1.x * P1.row(2) - P1.row(0);
    cv::Mat temp2 = pt1.y * P1.row(2) - P1.row(1);
    temp1.copyTo(A1.row(0));
    temp2.copyTo(A1.row(1));
    // cout << "A1 \n" << A1 << endl;

    cv::Mat A2 = cv::Mat::zeros(2, 4, CV_64F);
    temp1 = pt2.x * P2.row(2) - P2.row(0);
    temp2 = pt2.y * P2.row(2) - P2.row(1);
    temp1.copyTo(A2.row(0));
    temp2.copyTo(A2.row(1));
    // cout << "A2 \n" << A2 << endl;

    // stack -> svd > X.world -> X1 = X.world, X2 = R,t Mat * X.world -> X1.Z > 0 && X2.Z > 0
    // A is 4x4
    cv::Mat A;
    vconcat(A1, A2, A);

    // AX = 0, get X, x = 3x1
    cv::SVD aSVD(A);
    cv::Mat X1 = aSVD.vt.row(aSVD.vt.rows - 1).t();
    X1 = X1 / X1.at<double>(3, 0);
    cv::Mat X2 = Rt * X1;
    X2 = X2 / X2.at<double>(3, 0);
    // cout << "X1: \n" << X1 << endl;
    // cout << "X2: \n" << X2 << endl;

    bool all_positive = X1.at<double>(2, 0) > 0 && X2.at<double>(2, 0) > 0;

    return all_positive;
}

void point2DTo3D(cv::Mat Rt, vector<cv::Point2f> pts_1, vector<cv::Point2f> pts_2, cv::Mat P1, cv::Mat P2, cv::Mat &points3D)
// if use Point3D datastructure - void point2DTo3D(cv::Mat Rt, vector<cv::Point2f> pts_1, vector<cv::Point2f> pts_2, cv::Mat P1, cv::Mat P2, vector<cv::Point3d> &points3D) 
{ 
    for (size_t idx = 0; idx < pts_1.size(); idx++)
    {
        cv::Point2f pt1 = pts_1[idx];
        cv::Point2f pt2 = pts_2[idx];

        // Get A matrixes for both points, P1
        cv::Mat A1 = cv::Mat::zeros(2, 4, CV_64F);
        cv::Mat temp1 = pt1.x * P1.row(2) - P1.row(0);
        cv::Mat temp2 = pt1.y * P1.row(2) - P1.row(1);
        temp1.copyTo(A1.row(0));
        temp2.copyTo(A1.row(1));

        cv::Mat A2 = cv::Mat::zeros(2, 4, CV_64F);
        temp1 = pt2.x * P2.row(2) - P2.row(0);
        temp2 = pt2.y * P2.row(2) - P2.row(1);
        temp1.copyTo(A2.row(0));
        temp2.copyTo(A2.row(1));

        cv::Mat A;
        vconcat(A1, A2, A);

        cv::SVD aSVD(A);
        cv::Mat X1 = aSVD.vt.row(aSVD.vt.rows - 1).t();

        for (size_t i = 0; i < 4; i++)
        {
            points3D.at<double>(idx, i) = X1.at<double>(i, 0);
        }
        // // if use Point3D datastructure
        // points3D.push_back(Point3d(X1.at<double>(0, 0) / X1.at<double>(3, 0), X1.at<double>(1, 0) / X1.at<double>(3, 0), X1.at<double>(2, 0) / X1.at<double>(3, 0) ));
    }
}

void FindTriangulation(cv::Mat E, cv::Mat K, vector<cv::Point2f> prev_pts, vector<cv::Point2f> next_pts, cv::Mat &R, cv::Mat &t)
{
    cv::Mat W = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat Z = cv::Mat::zeros(3, 3, CV_64F);
    W.at<double>(0, 1) = -1;
    W.at<double>(1, 0) = 1;
    W.at<double>(2, 2) = 1;
    Z.at<double>(0, 1) = 1;
    Z.at<double>(1, 0) = -1;

    // SVD E to get S and R
    cv::SVD e_pre_SVD(E); // get eSVD.vt, eSVD.u;
    cv::Mat newE = e_pre_SVD.u * (Z * W) * e_pre_SVD.vt;
    cv::SVD eSVD(newE); // get eSVD.vt, eSVD.u;

    // S1, R1, S2, R2
    cv::Mat S1 = -eSVD.u * Z * eSVD.u.t();
    cv::Mat S2 = eSVD.u * Z * eSVD.u.t();
    cv::Mat R1 = eSVD.u * W.t() * eSVD.vt;
    cv::Mat R2 = eSVD.u * W * eSVD.vt;
    if (cv::determinant(R1) < 0)
        R1 = -R1;
    if (cv::determinant(R2) < 0)
        R2 = -R2;
    // SVD St = 0, get two possible t values
    cv::SVD s1SVD(S1); // get eSVD.vt, eSVD.u;
    cv::Mat t1 = s1SVD.vt.row(s1SVD.vt.rows - 1);
    cv::Mat t2 = -t1;

    // find K projection
    Mat extra_zeros = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat K_proj;
    cv::hconcat(K, extra_zeros, K_proj);

    // find projections 1 and 2
    cv::Mat P1 = K_proj * cv::Mat::eye(4, 4, CV_64F);
    cv::Mat Rt_1;
    cv::Mat Rt_2;
    cv::Mat Rt_3;
    cv::Mat Rt_4;
    cv::Mat P2_1;
    cv::Mat P2_2;
    cv::Mat P2_3;
    cv::Mat P2_4;
    FindProjection(R1, t1, K_proj, Rt_1, P2_1);
    FindProjection(R1, t2, K_proj, Rt_2, P2_2);
    FindProjection(R2, t1, K_proj, Rt_3, P2_3);
    FindProjection(R2, t2, K_proj, Rt_4, P2_4);
    int count[4] = {0, 0, 0, 0};
    // iterate through points, count which is better
    for (size_t idx = 0; idx < prev_pts.size(); idx++)
    {
        bool is_valid_1 = validateProjection(Rt_1, prev_pts[idx], next_pts[idx], P1, P2_1);
        bool is_valid_2 = validateProjection(Rt_2, prev_pts[idx], next_pts[idx], P1, P2_2);
        bool is_valid_3 = validateProjection(Rt_3, prev_pts[idx], next_pts[idx], P1, P2_3);
        bool is_valid_4 = validateProjection(Rt_4, prev_pts[idx], next_pts[idx], P1, P2_4);

        // cout << "one: " << is_valid_1 << "; two: " << is_valid_2 << "; three: " << is_valid_3 << "; four: " << is_valid_4 << endl;

        if (is_valid_1)
            count[0]++;
        if (is_valid_2)
            count[1]++;
        if (is_valid_3)
            count[2]++;
        if (is_valid_4)
            count[3]++;
    }

    // cout << "count: \n" << count[0] << " " << count[1] << " " << count[2] << " " << count[3]<< endl;

    const int N = sizeof(count) / sizeof(int);
    const int P_max = distance(count, max_element(count, count + N));
    // cout << "Index of selected P " << P_max << endl;

    // iterate through points, find the points
    cv::Mat P2;
    cv::Mat Rt;
    vector<Point3d> points3D;

    switch (P_max)
    {
    case 0:
        Rt = Rt_1;
        P2 = P2_1;
        break;
    case 1:
        Rt = Rt_2;
        P2 = P2_2;
        break;
    case 2:
        Rt = Rt_3;
        P2 = P2_3;
        break;
    case 3:
        Rt = Rt_4;
        P2 = P2_4;
        break;
    default:
        Rt = Rt_1;
        P2 = P2_1;
        cout << "Error: incorrect P2 value" << endl;
    }

    int points_size = prev_pts.size();
    cv::Mat points3DMat = cv::Mat::zeros(prev_pts.size(), 4, DataType<double>::type);
    point2DTo3D(Rt, prev_pts, next_pts, P1, P2, points3DMat);

    // ========== Start Testing 3D Points ===========
    cv::Mat points4D_cvt(4, points_size, CV_64FC1);
    cv::Mat points3D_cv = cv::Mat::zeros(points_size, 3, DataType<double>::type);
    cv::triangulatePoints(P1, P2, prev_pts, next_pts, points4D_cvt);
    cv::Mat points4D_cv = points4D_cvt.t();

    cout << "4D Points \n " << points3DMat << endl;
    cout << "4D Points CV \n " << points4D_cv << endl;
    // ========== End Testing 3D Points ===========

    R = Rt(Rect(0, 0, 3, 3));
    t = Rt(Rect(3, 0, 1, 3));
}

void testResult()
{
    cv::Mat points3D(1, 16, CV_64FC4);
    // cv::randu(points3D, cv::Scalar(-5.0, -5.0, 1.0, 1.0), cv::Scalar(5.0, 5.0, 10.0, 1.0 ));
    cv::randu(points3D, cv::Scalar(100.0, 100.0, 1.0, 1.0), cv::Scalar(400, 400, 1.0, 1.0));
    cv::Mat norm = (Mat_<double>(3, 3) << 0.001, 0, 0, 0, 0.001, 0, 0, 0, 1);

    //Compute 2 camera matrices
    cv::Matx34d C1 = cv::Matx34d::eye();
    cv::Matx34d C2 = cv::Matx34d::eye();
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);

    // Translation
    C2(0, 3) = 100;
    // C2(2, 3) = 1;

    // Rotation
    // C2(0,0) = 0;
    // C2(0,1) = -1;
    // C2(1,0) = 1;
    // C2(1,1) = 0;

    //Compute points projection
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (int i = 0; i < points3D.cols; i++)
    {
        cv::Vec3d hpt1 = C1 * points3D.at<cv::Vec4d>(0, i);
        cv::Vec3d hpt2 = C2 * points3D.at<cv::Vec4d>(0, i);
        hpt1 /= hpt1[2];
        hpt2 /= hpt2[2];

        cv::Point2f p1(hpt1[0], hpt1[1]);
        cv::Point2f p2(hpt2[0], hpt2[1]);

        points1.push_back(p1);
        points2.push_back(p2);
    }
    // cv::Matx33d F_cv = (Matx33d)cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 1.5f);
    cv::Matx33d F = Findfundamental(points1, points2, 1, 1, norm);
    cv::Mat E = FindEssential(F, K);
    // cv::Mat tx = cv::Mat::zeros(3,3,CV_64F);
    // tx.at<double>(0,1)= - C2(2, 3);
    // tx.at<double>(0,2)= C2(1, 3);
    // tx.at<double>(1,0)= C2(2, 3);
    // tx.at<double>(1,2)= - C2(0, 3);
    // tx.at<double>(2,0)= - C2(1, 3);
    // tx.at<double>(2,1)= C2(0, 3);
    // cv::Mat Rx = cv::Mat::eye(3,3,CV_64F);
    // Rx.at<double>(0,0)= C2(0, 0);
    // Rx.at<double>(0,1)= C2(0, 1);
    // Rx.at<double>(1,0)= C2(1, 0);
    // Rx.at<double>(1,1)= C2(1, 1);
    // cv::Mat E = tx*Rx;
    cv::Mat R;
    cv::Mat t;
    FindTriangulation(E, K, points1, points2, R, t);

    //Print
    std::cout << "C1 \n " << C1 << std::endl;
    std::cout << "C2 \n " << C2 << std::endl;
    std::cout << "R \n " << R << std::endl;
    std::cout << "t \n " << t << std::endl;
    Mat img_1(480, 640, CV_8UC3, cv::Scalar(0, 0, 255));
    Mat img_2(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
    drawEpipolarLines(F, img_1, img_2, points1, points2, 1.5f);
}

int main(int argc, char **argv)
{

    srand(time(NULL));

    if (argc != 3)
    {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    // Get input img size
    int img_w = img_1.size().width;
    int img_h = img_1.size().height;

    list<cv::Point2f> keypoints;
    vector<cv::KeyPoint> kps;

    // OpenCV2
    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector> detector = Algorithm::create<FeatureDetector>(detectorType);
    detector->set("thres", 100);

    // OpenCV3
    // Ptr<BRISK> detector = BRISK::create(100);

    detector->detect(img_1, kps);

    for (auto kp : kps)
        keypoints.push_back(kp.pt);

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for (auto kp : keypoints)
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img_1, img_2, prev_keypoints, next_keypoints, status, error);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "LK Flow use time：" << time_used.count() << " seconds." << endl;

    vector<cv::Point2f> kps_prev, kps_next;
    kps_prev.clear();
    kps_next.clear();
    for (size_t i = 0; i < prev_keypoints.size(); i++)
    {
        if (status[i] == 1)
        {
            kps_prev.push_back(prev_keypoints[i]);
            kps_next.push_back(next_keypoints[i]);
        }
    }

    // p Probability that at least one valid set of inliers is chosen
    // d Tolerated distance from the model for inliers
    // e Assumed outlier percent in data set.
    double p = 0.99;
    double d = 1.5f;
    double e = 0.2;

    int niter = static_cast<int>(std::ceil(std::log(1.0 - p) / std::log(1.0 - std::pow(1.0 - e, 8))));
    // normalize the points
    cv::Mat norm = (Mat_<double>(3, 3) << 3.0 / img_h, 0, -1.5, 0, 3.0 / img_h, -1.5, 0, 0, 1.2);
    Mat Fundamental;
    cv::Matx33d F, Fcandidate;
    int bestinliers = -1;
    vector<cv::Point2f> prev_subset, next_subset;
    int matches = kps_prev.size();
    prev_subset.clear();
    next_subset.clear();

    for (int i = 0; i < niter; i++)
    {
        // step1: randomly sample 8 matches for 8pt algorithm
        unordered_set<int> rand_util;
        while (rand_util.size() < 8)
        {
            int randi = rand() % matches;
            rand_util.insert(randi);
        }
        vector<int> random_indices(rand_util.begin(), rand_util.end());
        for (size_t j = 0; j < rand_util.size(); j++)
        {
            prev_subset.push_back(kps_prev[random_indices[j]]);
            next_subset.push_back(kps_next[random_indices[j]]);
        }
        // step2: perform 8pt algorithm, get candidate F

        Fcandidate = Findfundamental(prev_subset, next_subset, img_w, img_h, norm);

        // step3: Evaluate inliers, decide if we need to update the best solution
        int inliers = 0;
        for (size_t j = 0; j < kps_prev.size(); j++)
        {
            if (checkinlier(prev_keypoints[j], next_keypoints[j], Fcandidate, d))
            {
                inliers++;
            }
        }

        if (inliers > bestinliers)
        {
            F = Fcandidate;
            bestinliers = inliers;
        }
        prev_subset.clear();
        next_subset.clear();
    }

    // step4: After we finish all the iterations, use the inliers of the best model to compute Fundamental matrix again.
    for (size_t j = 0; j < kps_prev.size(); j++)
    {
        if (checkinlier(kps_prev[j], kps_next[j], F, d))
        {
            prev_subset.push_back(kps_prev[j]);
            next_subset.push_back(kps_next[j]);
        }
    }
    cout << kps_next.size() << endl;
    cout << next_subset.size() << endl;

    // // ============= Start Testing FindFundamental ==============

    F = Findfundamental(prev_subset, next_subset, img_w, img_h, norm);

    cout << "Our Fundamental is \n"
         << F << endl;

    cv::Matx33d F_cv;
    F_cv = (Matx33d)cv::findFundamentalMat(prev_subset, next_subset, cv::FM_8POINT, 1.5f);

    cout << "OpenCV Fundamental is \n"
         << F_cv << endl;

    drawEpipolarLines(F, img_1,img_2, prev_subset, next_subset, d);
    drawEpipolarLines(F_cv, img_1, img_2, prev_subset, next_subset, d);
    // ============= End Testing FindFundamental ==============

    // ============= Start Testing FindTriangulation ==============
    double camera_fx = 517.3;
    double camera_fy = 516.5;
    double camera_cx = 318.643040;
    double camera_cy = 255.313989;

    cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
    K.at<double>(0, 0) = camera_fx;
    K.at<double>(1, 1) = camera_fy;
    K.at<double>(0, 2) = camera_cx;
    K.at<double>(1, 2) = camera_cy;
    K.at<double>(2, 2) = 1;

    cv::Mat E;
    E = FindEssential(F_cv, K);

    cv::Mat R;
    cv::Mat t;
    FindTriangulation(E, K, kps_prev, kps_next, R, t);
    cout << "Our Rotation matrix is \n"
         << R << endl;
    cout << "Our Translation matrix is \n"
         << t << endl;

    cv::Mat R_cv;
    cv::Mat t_cv;
    cv::Point2d center(camera_cx, camera_cy);
    // recoverPose(E, prev_subset, next_subset, R_cv, t_cv, camera_fx, center);
    // cout << "OpenCV Rotation matrix is \n" << R_cv << endl;
    // cout << "OpenCV Translation matrix is \n" << t_cv << endl;

    // ============= End Testing FindTriangulation ==============

    testResult();
}
