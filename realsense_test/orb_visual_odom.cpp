#include <iostream>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include <chrono>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2d(
    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
  );
}

int main(int argc, char *argv[]){
	rs2::pipeline realsense_pip;
	rs2::config config;
	config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);
	config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
	//config.enable_stream(RS2_STREAM_INFRARED,640,480,RS2_FORMAT_Y8,15);
	
	rs2::pipeline_profile pipeProfile = realsense_pip.start(config);
	auto sensor = pipeProfile.get_device().first<rs2::depth_sensor>();
	auto scale =  sensor.get_depth_scale();
	rs2_intrinsics intrinsics = pipeProfile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
	
	rs2::frameset frames;
	std::vector<cv::cuda::GpuMat> Mat_buffer;
	std::vector<cv::Mat> Mat_depth_buffer;
	for(int i = 0; i<10;i++){
					frames = realsense_pip.wait_for_frames();
}
	while(true){
		for(int i = 0; i<2;i++){
			frames = realsense_pip.wait_for_frames();
			rs2::frame color_frame = frames.get_color_frame();
			cv::Mat color(cv::Size(640,480),CV_8UC3,(void*)color_frame.get_data(),cv::Mat::AUTO_STEP);
			cv::Mat depth(cv::Size(640,480),CV_16UC1,(void*)color_frame.get_data(),cv::Mat::AUTO_STEP);
			cv::Mat wb_img;
			cv::cvtColor(color,wb_img,cv::COLOR_BGR2GRAY);
			cv::cuda::GpuMat wb_now(wb_img);
			Mat_buffer.push_back(wb_now);
			Mat_depth_buffer.push_back(depth);
		}
		std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
		cv::cuda::GpuMat descriptors_1,descriptors_2;
		cv::Mat K = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.ppx, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 1);
		
		cv::Ptr<cv::FeatureDetector> detector = cv::cuda::ORB::create(100);
		cv::Ptr<cv::DescriptorExtractor> descriptor = cv::cuda::ORB::create();
		cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
		//cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
		
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		detector -> detect(Mat_buffer[0],keypoints_1);
		detector -> detect(Mat_buffer[1],keypoints_2);
		
		descriptor -> compute(Mat_buffer[0],keypoints_1,descriptors_1);
		descriptor -> compute(Mat_buffer[1],keypoints_2,descriptors_2);
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
		std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		std::cout<<"extract ORB cost: "<<time_used.count()<<std::endl;
		
		std::vector<cv::DMatch> matches;
		t1 = std::chrono::steady_clock::now();
		matcher -> match(descriptors_1,descriptors_2,matches);
		t2 = std::chrono::steady_clock::now();
		time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);	
		std::cout<<"match ORB cost: "<<time_used.count()<<std::endl;
		
		
		auto min_max = std::minmax_element(matches.begin(),matches.end(),[](const cv::DMatch &m1, const cv::DMatch &m2){return m1.distance < m2.distance;});	
		double min_dist = min_max.first -> distance;
		double max_dist = min_max.second -> distance;
		
		std::vector<cv::DMatch> good_matches;
		for(int i= 0; i<descriptors_1.rows;i++){
			if(matches[i].distance <= std::max(2*min_dist,30.0)){
				good_matches.push_back(matches[i]);
				}
			}
		
		std::vector<cv::Point3f> pts1, pts2;
		std::cout<<"create 3d point clound "<<std::endl;
		for(cv::DMatch m:good_matches){
			ushort d1 = Mat_depth_buffer[0].ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
			ushort d2 = Mat_depth_buffer[1].ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_1[m.trainIdx].pt.x)];
			if (d1 == 0 || d2 == 0) 
			continue;
			cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
			cv::Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
			
			float dd1 = float(d1) * scale;
			float dd2 = float(d2) * scale;
			
			std::cout<<dd1<<std::endl;
			
			pts1.push_back(cv::Point3f(p1.x * dd1, p1.y * dd1, dd1));
			pts2.push_back(cv::Point3f(p2.x * dd2, p2.y * dd2, dd2));
			}
		  std::cout << "3d-3d pairs: " << pts1.size() << std::endl;
		  
		  //////svd//////////
		  cv::Point3f p1,p2;
		  int N = pts1.size();
		  for(int i = 0;i < N; i++){
			  p1 += pts1[i];
			  p2 += pts2[i];
			  }
		  p1 = cv::Point3f(cv::Vec3f(p1)/N);
		  p2 = cv::Point3f(cv::Vec3f(p2)/N);
			  
		  std::vector<cv::Point3f> q1(N),q2(N);
		  for(int i = 0; i < N; i ++){
			   q1[i] = pts1[i] - p1;
			   q2[i] = pts2[i] - p2;
			   }
		   Eigen:: Matrix3d W = Eigen::Matrix3d::Zero();
		   
		   for (int i = 0; i < N; i ++){
			   W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z).transpose();
			   }
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Eigen::Matrix3d U = svd.matrixU();
			Eigen::Matrix3d V = svd.matrixV();
			
			Eigen::Matrix3d R_ = U * (V.transpose());
			if (R_.determinant() < 0) {
				R_ = -R_;
			}
			Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
			
		   cv::Mat R = (cv::Mat_<double>(3,3)<<R_(0,0),R_(0,1),R_(0,2),R_(1,0),R_(1,1),R_(1,2),R_(2,0),R_(2,1),R_(2,2));
		   cv::Mat t = (cv::Mat_<double>(3,1)<<t_(0,0), t_(1,0),t_(2,0));
		   std::cout<<"t: "<<t_(0,0)<<" "<<t_(1,0)<<" "<<t_(2,0)<<std::endl;
			
		//cv::Mat img_goodmatch;
		//cv::Mat img_0;
		//cv::Mat img_1;
		//Mat_buffer[0].download(img_0);
		//Mat_buffer[1].download(img_1);
		
	//	cv::drawMatches(img_0,keypoints_1,img_1,keypoints_2,good_matches, img_goodmatch);
		

	//	cv::imshow("good_match",img_goodmatch);


	//	char c = cv::waitKey(1);
	//	if (c =='s'){}
	//	else if (c == 'q'){
	//		break;
	//		}
		Mat_buffer.clear();
		char c=' ';
		while (c !='q'){
			c = cv::waitKey(1);
			}


	}
	return 0;
	}
