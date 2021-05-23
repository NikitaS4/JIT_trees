cmake_minimum_required(VERSION 3.4...3.18)
project(regbm)


# add definitions for xtensor
#add_definitions(-DXTENSOR_ENABLE_XSIMD)
#add_definitions(-DXTENSOR_USE_XSIMD)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx2 -ffast-math")
set(CMAKE_BUILD_TYPE "Release")

# include pybind11
# this will also import pybind11_add_module function
find_package(pybind11 REQUIRED)

# include all directories needed to import xtensor
# NOTE: theese paths are platform-specific
# Windows

# To find a path to numpy installation, You can launch in Python:
# import numpy as np
# print(np.get_include())

include_directories(D:/anaconda3/lib/site-packages/numpy/core/include)
include_directories(D:/anaconda3/lib/site-packages/pybind11/include)
include_directories(D:/anaconda3/include)
include_directories(D:/anaconda3/Library/include)

# On Linux:

#include_directories(/home/__user__/.local/lib/python3.8/site-packages/numpy/core/include)
#include_directories(/home/__user__/.local/lib/python3.8/site-packages/pybind11/include)
#include_directories(/home/__user__/include)


# get all *.cpp files recursively
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h)
# like add_executable but for pybind11 module
pybind11_add_module(regbm ${SRC_FILES})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(regbm PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
