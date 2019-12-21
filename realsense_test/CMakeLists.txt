cmake_minimum_required(VERSION 3.10.2)

project(realsense_test)
find_package(realsense2 REQUIRED)
add_executable(realsense_test realsense_test.cpp)
target_link_libraries(realsense_test ${DEPENDENCIES} realsense2)

