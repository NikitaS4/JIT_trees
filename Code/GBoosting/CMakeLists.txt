cmake_minimum_required(VERSION 3.4...3.18)
project(JIT_trees)


# add definitions for xtensor
#add_definitions(-DXTENSOR_ENABLE_XSIMD)
#add_definitions(-DXTENSOR_USE_XSIMD)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx2 -ffast-math")

# include pybind11
# this will also import pybind11_add_module function
find_package(pybind11 REQUIRED)
include_directories(D:/anaconda3/lib/site-packages/numpy/core/include)
# get all *.cpp files recursively
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h)
# like add_executable but for pybind11 module
pybind11_add_module(pybind11 ${SRC_FILES})#src/pybind/pybind.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(pybind11 PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

#include_directories(/home/--user--/include)
#include_directories(/home/--user--/.miniconda3/lib/python3.7/site-packages/numpy/core/include)
