#ifndef GBOOSTING_H
#define GBOOSTING_H
#pragma once

#include "Structs.h"
#include "GBHist.h"
#include "GBDecisionTree.h"
#include "History.h"
#include <vector>


class GradientBoosting {
public:
	GradientBoosting(const size_t binCount = defaultBinCount,
					 const size_t patience = defaultPatience);
	virtual ~GradientBoosting();
	// 1st dim - object number, 2nd dim - feature number
	// fit return the number of estimators (include constant estim)
	History fit(const std::vector<std::vector<FVal_t>>& xTrain, 
				const std::vector<Lab_t>& yTrain,
				const std::vector<std::vector<FVal_t>>& xValid,
				const std::vector<Lab_t>& yValid,
				const size_t treeCount,
				const size_t treeDepth,
				const float learningRate = defaultLR);
	Lab_t predict(const std::vector<FVal_t>& xTest) const;

	void printModel() const;
protected:
	static std::vector<size_t> sortFeature(const std::vector<FVal_t>& xData,
		std::vector<size_t>& backIdxs);
	void swapAxes(const std::vector<std::vector<FVal_t>>& xTrain);
	static Lab_t loss(const std::vector<Lab_t>& pred, 
					  const std::vector<Lab_t>& truth);
	inline bool canStop(const size_t stepNum) const;

	// fields
	size_t featureCount;
	size_t trainLen;
	size_t realTreeCount;
	size_t binCount;
	size_t patience;
	// xSwapped: 1st dim - feature number, 2nd dim - object number
	std::vector<std::vector<FVal_t>> xSwapped;
	std::vector<GBHist> hists; // histogram for each feature
	std::vector<GBDecisionTree> trees;
	Lab_t zeroPredictor; // constant model
	std::vector<Lab_t> trainLosses;
	std::vector<Lab_t> validLosses;

	// constants
	static const size_t defaultBinCount = 128;
	static const float defaultLR;
	static const size_t defaultPatience = 3;
};

#endif // GBOOSTING_H

