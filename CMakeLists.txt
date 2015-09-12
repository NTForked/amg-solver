project(amg-solver)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-fopenmp -fpermissive -std=c++0x ${CMAKE_CXX_FLAGS}")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")

include(geo_sim_sdk)
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

add_library(amg SHARED amg_solver.cc coarsener.cc linear_solver.cc smoother.cc)
target_link_libraries(amg
    ${UMFPACK_LIBRARIES}
)

add_executable(amg_test main.cc)
target_link_libraries(amg_test
    amg
    zjucad-ptree
    hj-util
)

add_executable(solve_harmonic harmonic.cc)
target_link_libraries(solve_harmonic
    amg
    jtf-mesh
)
