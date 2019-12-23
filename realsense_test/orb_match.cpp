#include <iostream>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include <chrono>


int main(int argc, char *argv[]){
	rs2::pipeline realsense_pip;
	rs2::config config;
	config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);
	//config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,15);
	//config.enable_stream(RS2_STREAM_INFRARED,640,480,RS2_FORMAT_Y8,15);
	

	realsense_pip.start(config);
	rs2::frameset frames;
	std::vector<cv::cuda::GpuMat> Mat_buffer;
	for(int i = 0; i<30;i++){
					frames = realsense_pip.wait_for_frames();
}
	while(true){
		for(int i = 0; i<2;i++){
			frames = realsense_pip.wait_for_frames();
			rs2::frame color_frame = frames.get_color_frame();
			cv::Mat color(cv::Size(640,480),CV_8UC3,(void*)color_frame.get_data(),cv::Mat::AUTO_STEP);
			cv::Mat wb_img;
			cv::cvtColor(color,wb_img,cv::COLOR_BGR2GRAY);
			cv::cuda::GpuMat wb_now(wb_img);
			Mat_buffer.push_back(wb_now);
		}
		std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
		cv::cuda::GpuMat descriptors_1,descriptors_2;
		
		cv::Ptr<cv::FeatureDetector> detector = cv::cuda::ORB::create();
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
		
		cv::Mat img_match;
		cv::Mat img_goodmatch;
		cv::Mat img_0;
		cv::Mat img_1;
		Mat_buffer[0].download(img_0);
		Mat_buffer[1].download(img_1);
		
		cv::drawMatches(img_0,keypoints_1,img_1,keypoints_2,matches, img_match);
		cv::drawMatches(img_0,keypoints_1,img_1,keypoints_2,good_matches, img_goodmatch);
		
		cv::imshow("all_matches",img_match);
		cv::imshow("good_match",img_goodmatch);


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
