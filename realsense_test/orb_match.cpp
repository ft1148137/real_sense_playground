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
	std::cout<<"/n set_config";
	config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,15);
	config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,15);
	config.enable_stream(RS2_STREAM_INFRARED,640,480,RS2_FORMAT_Y8,15);

	realsense_pip.start(config);
	rs2::frameset frames;
	std::vector<cv::cuda::GpuMat> Mat_buffer;

	while(true){
		for(int i = 0; i<2;i++){
			frames = realsense_pip.wait_for_frames();
			rs2::frame color_frame = frames.get_color_frame();
			cv::Mat color(cv::Size(640,480),CV_8UC3,(void*)color_frame.get_data(),cv::Mat::AUTO_STEP);
			cv::cuda::GpuMat color_now(color);
			Mat_buffer.push_back(color_now);
		}
		std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
		cv::cuda::GpuMat descriptors_1,descriptors_2;
		
		cv::Ptr<cv::FeatureDetector> detector = cv::cuda::ORB::create();
		cv::Ptr<cv::DescriptorExtractor> descriptor = cv::cuda::ORB::create();
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForece-Hamming");
		
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		detector -> detect(Mat_buffer[0],keypoints_1);
		detector -> detect(Mat_buffer[1],keypoints_2);
		
		descriptor -> compute(Mat_buffer[0],keypoints_1,descriptors_1);
		descriptor -> compute(Mat_buffer[1],keypoints_2,descriptors_2);
		std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

				
		cv::Mat mat_plot;
		Mat_buffer[0].download(mat_plot);
		cv::imshow("Display_color",mat_plot);
		char c = cv::waitKey(1);
		if (c =='s'){}
		else if (c == 'q'){
			break;
			}
		Mat_buffer.clear();
	//	char c=' ';
	//	while (c !='q'){
	//		c = cv::waitKey(1);
	//		}

		}
	return 0;
	}
