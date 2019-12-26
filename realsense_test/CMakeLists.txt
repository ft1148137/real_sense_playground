cmake_minimum_required(VERSION 3.10.2)

project(realsense_test)
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(g2o REQUIRED)
add_executable(realsense_test realsense_test.cpp)
add_executable(orb_match orb_match.cpp)
add_executable(orb_visual_odom orb_visual_odom.cpp)
target_link_libraries(realsense_test ${DEPENDENCIES} realsense2 ${OpenCV_LIBS})
target_link_libraries(orb_match ${DEPENDENCIES} realsense2 ${OpenCV_LIBS})
target_link_libraries(orb_visual_odom ${DEPENDENCIES} realsense2 ${OpenCV_LIBS} Eigen3::Eigen)

