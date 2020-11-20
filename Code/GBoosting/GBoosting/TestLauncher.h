#pragma once

#include <iostream>
#include <vector>

#include "GBoosting.h"
#include "GBTest.h"
#include "StatisticsHelper.h"
#include "DataLoader.h"


class TestLauncher {
public:
	TestLauncher(const std::vector<std::vector<FVal_t>>& xTrain,
		const std::vector<Lab_t>& yTrain,
		const std::vector<std::vector<FVal_t>>& xValid,
		const std::vector<Lab_t>& yValid);
	
	// init with files contain the input data
	TestLauncher(const std::string& xTrainFile,
		const std::string& yTrainFile,
		const std::string& xValidFile,
		const std::string& yValidFile);

	void performTest(const std::vector<size_t>& treeCounts,
		const std::vector<size_t>& treeDepths,
		const std::vector<size_t>& binCounts,
		const std::vector<float>& learnRates) const;

	void singleTestPrint(const size_t treeCount,
		const size_t treeDepth, const size_t binCount,
		const float learnRate) const;
private:
	// fields
	std::vector<std::vector<FVal_t>> xTrain;
	std::vector<Lab_t> yTrain;
	std::vector<std::vector<FVal_t>> xValid;
	std::vector<Lab_t> yValid;

	// functions
	static inline void printConditions(const size_t treeNum,
		const size_t depth, const size_t binCount,
		const float learningRate);

	std::vector<Lab_t> computeResiduals(const GradientBoosting& model) const;

	TestLauncher() = delete;  // can't create without train&validation data
};