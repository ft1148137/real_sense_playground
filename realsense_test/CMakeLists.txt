cmake_minimum_required(VERSION 3.10.2)

project(realsense_test)
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(realsense_test realsense_test.cpp)
add_executable(orb_match orb_match.cpp)
target_link_libraries(realsense_test ${DEPENDENCIES} realsense2 ${OpenCV_LIBS})
target_link_libraries(orb_match ${DEPENDENCIES} realsense2 ${OpenCV_LIBS})

