cmake_minimum_required(VERSION 3.14)

project(JITtrees)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME} "Main.cpp"
	"Structs.h"
	"GBoosting.h" "GBoosting.cpp"
	"GBDecisionTree.h" "GBDecisionTree.cpp"
	"DataLoader.h" "DataLoader.cpp"
	"GBHist.h" "GBHist.cpp"
	"History.h" "History.cpp"
	"StatisticsHelper.h" "StatisticsHelper.cpp"
	"TestLauncher.h" "TestLauncher.cpp")
