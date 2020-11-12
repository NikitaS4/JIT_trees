#pragma once

#include "Structs.h"
#include "GBHist.h"
#include "GBDecisionTree.h"
#include <vector>

class GradientBoosting {
public:
	GradientBoosting();
	virtual ~GradientBoosting();
	// 1st dim - object number, 2nd dim - feature number
	void fit(const std::vector<std::vector<FVal_t>>& xTrain, 
			 const std::vector<Lab_t>& yTrain, 
			 const size_t treeCount);
	Lab_t predict(const std::vector<FVal_t>& xTest);
protected:
	static void sortFeature(const std::vector<FVal_t>& xData, 
		std::vector<size_t>& sortedIdxs);
	void swapAxes(const std::vector<std::vector<FVal_t>>& xTrain);

	// fields
	size_t featureCount;
	size_t trainLen;
	size_t realTreeCount;
	size_t binCount;
	// xSwapped: 1st dim - feature number, 2nd dim - object number
	std::vector<std::vector<FVal_t>> xSwapped;
	std::vector<GBHist> hists; // histogram for each feature
	std::vector<GBDecisionTree> trees;
	Lab_t zeroPredictor; // constant model

	// constants
	static const size_t defaultBinCount = 8;
};