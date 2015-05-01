project(amg-solver)
cmake_minimum_required(VERSION 2.8)

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif(EIGEN3_FOUND)

file(GLOB header *.h)
file(GLOB source *.cc)

add_library(amg SHARED ${header} ${source})
