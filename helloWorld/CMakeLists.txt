cmake_minimum_required(VERSION 2.8)

project(helloWorld)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# helloworld
add_executable(helloWorld helloworld.cc)
target_link_libraries(helloWorld ${CERES_LIBRARIES})

set(CMAKE_BUILD_TYPE Debug)