#include <fstream>
#include <string>
#include "logic/VisualOdometryLogic.h"
#include "backend/BackendSFM.h"
#include "logic/config.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pangolin/pangolin.h>
//#include "frontend/idscam.h"



#include<chrono>

using namespace cv;
using namespace std;
using namespace cv::gpu;

void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc, float camerasize, float mCameraLineWidth)
{
    const float &w = camerasize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
    glMultMatrixf(Twc.m);
#else
    glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}





int main(int argc, char *argv[])
{
    myslam::Config::setParameterFile ( argv[1] );

    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/rgb.txt" );
    if ( !fin )
    {
        cout<<"can not find rgb.txt!"<<endl;
        return 1;
    }

    vector<string> rgb_files;
    vector<double> rgb_times;
    string buff;
// skip header
    for(int i=0;i<9;i++)
    {
	fin >> buff;
    }
// get rgb filenames
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file;
        fin >> rgb_time >> rgb_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        rgb_files.push_back(dataset_dir + "/" + rgb_file);
        if ( fin.good() == false )
            break;
    }


	BackendSFM backend;

	VisualFrontend frontend;

	double fx,fy,cx,cy;
	fx = myslam::Config::get<float>("camera.fx");
	fy = myslam::Config::get<float>("camera.fy");
	cx = myslam::Config::get<float>("camera.cx");
	cy = myslam::Config::get<float>("camera.cy");

	VisualOdometryLogic logic(frontend, backend, fx, fy, cx, cy);

	// set up visualizer
	pangolin::CreateWindowAndBind("Simple odometry: Camera motion",1024,768);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
	pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", false ,true);
	pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
	pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
	pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
	pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode",false,true);
	pangolin::Var<bool> menuReset("menu.Reset",false,false);


	float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

	mViewpointX = myslam::Config::get<float>("ViewpointX");
	mViewpointY = myslam::Config::get<float>("ViewpointY");
	mViewpointZ = myslam::Config::get<float>("ViewpointZ");
	mViewpointF = myslam::Config::get<float>("ViewpointF");

	cout<<"mViewpointX: "<<mViewpointX<<" "<<mViewpointY<<" "<<mViewpointZ<<" "<<mViewpointF<<endl;

	pangolin::OpenGlRenderState s_cam(
	    pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
	    pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
	);

	// Add named OpenGL viewport to window and provide 3D Handler
	pangolin::View& d_cam = pangolin::CreateDisplay()
	    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
	    .SetHandler(new pangolin::Handler3D(s_cam));


	pangolin::OpenGlMatrix Twc;
	Twc.SetIdentity();


	// Loop all the frames

	cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
	size_t i=0;
	auto start = chrono::steady_clock::now();
	while ( i < rgb_files.size()-1)
	{
		cout << "****** loop " << i << " ******" << endl;

		Mat color1 = cv::imread(rgb_files[i],CV_LOAD_IMAGE_COLOR);
		i++;
		logic.handleImage(color1);
		//Eigen::Affine3d M = logic.getcurrentpose();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		logic.GetCurrentOpenGLCameraMatrix(Twc);


		if(menuFollowCamera)
		{
		    s_cam.Follow(Twc);
		}

		d_cam.Activate(s_cam);
		glClearColor(1.0f,1.0f,1.0f,1.0f);
		DrawCurrentCamera(Twc, myslam::Config::get<float>("CameraSize"),myslam::Config::get<float>("CameraLineWidth"));


		pangolin::FinishFrame();


		cout<<"current frame pose: \n"<<Twc<<endl;

		cout<<endl;
	}
	auto end = chrono::steady_clock::now();
	cout << "total running time: "<< chrono::duration <double, milli> (end-start).count() << " ms" << endl;
	return 0;
}
