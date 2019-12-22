#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <iostream>             // for cout
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>

int main (int argc, char *argv[])
{	
	rs2::pipeline realsense_pip;
	rs2::config config;
	
	config.enable_stream(RS2_STREAM_INFRARED,640,480,RS2_FORMAT_Y8,15);
	config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,15);
	config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,15);
	rs2::pipeline_profile pipeProfile = realsense_pip.start(config);
	rs2_intrinsics intrinsics = pipeProfile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
	rs2::frameset frames;
	std::cout<<"\n width: "<<intrinsics.width<<"\n height:"<<intrinsics.height<<"\n ppx: "<<intrinsics.ppx<<"\n ppy:"
			 <<intrinsics.ppy<<"\n fx: "<<intrinsics.fx<<"\n fy:"<<intrinsics.fy<<std::endl;
	while(true){
		frames = realsense_pip.wait_for_frames();
		rs2::frame ir_frame = frames.first(RS2_STREAM_INFRARED);
		rs2::frame depth_frame = frames.get_depth_frame();
		rs2::frame color_frame = frames.get_color_frame();
		cv::Mat ir(cv::Size(640,480),CV_8UC1,(void*)ir_frame.get_data(),cv::Mat::AUTO_STEP);
		cv::Mat color(cv::Size(640,480),CV_8UC3,(void*)color_frame.get_data(),cv::Mat::AUTO_STEP);
		cv::equalizeHist(ir,ir);
		cv::applyColorMap(ir,ir,cv::COLORMAP_JET);
		cv::imshow("Display_ir",ir);
		cv::imshow("Display_color",color);
		char c = cv::waitKey(1);
		if (c =='s'){}
		else if (c == 'q'){
			break;
			}
		}

	return 0;
}
