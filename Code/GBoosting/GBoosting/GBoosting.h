#pragma once

#include "Structs.h"
#include "GBHist.h"
#include "GBDecisionTree.h"
#include <vector>

class GradientBoosting {
public:
	GradientBoosting(const size_t binCount = defaultBinCount,
					 const size_t patience = defaultPatience);
	virtual ~GradientBoosting();
	// 1st dim - object number, 2nd dim - feature number
	// fit return the number of estimators (include constant estim)
	size_t fit(const std::vector<std::vector<FVal_t>>& xTrain, 
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
	inline bool canStop(const Lab_t trainLoss,
						const Lab_t validLoss,
						const size_t stepNum);

	// fields
	size_t featureCount;
	size_t trainLen;
	size_t realTreeCount;
	size_t binCount;
	size_t patience;
	bool* lossesRising;
	Lab_t lastLossDiff;
	// xSwapped: 1st dim - feature number, 2nd dim - object number
	std::vector<std::vector<FVal_t>> xSwapped;
	std::vector<GBHist> hists; // histogram for each feature
	std::vector<GBDecisionTree> trees;
	Lab_t zeroPredictor; // constant model

	// constants
	static const size_t defaultBinCount = 128;
	static const float defaultLR;
	static const size_t defaultPatience = 3;
};