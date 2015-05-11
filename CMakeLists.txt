project(amg-solver)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-fopenmp -fpermissive -std=c++0x ${CMAKE_CXX_FLAGS}")

include($ENV{HOME}/usr/share/cmake/Modules/geo_sim_sdk.cmake)
include_geo_sim_sdk()
link_geo_sim_sdk()
include_directories(${PROJECT_SOURCE_DIR})

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif(EIGEN3_FOUND)

find_package(UMFPACK REQUIRED)
if(UMFPACK_FOUND)
    include_directories(${UMFPACK_INCLUDES})
endif(UMFPACK_FOUND)

file(GLOB header *.h)
file(GLOB source *.cc)

add_library(amg SHARED ${header} ${source})
target_link_libraries(amg
    ${UMFPACK_LIBRARIES}
)

add_executable(test main.cc)
target_link_libraries(test
    amg
    zjucad-ptree
    hj-util
)
