#ifndef GBOOSTING_H
#define GBOOSTING_H

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
				const float learningRate = defaultLR,
				const Lab_t earlyStoppingDelta = defaultESDelta);
	Lab_t predict(const std::vector<FVal_t>& xTest) const;

	// predict "from-to" - predict using only subset of trees
	// first estimator - the first tree number to predict (enumeration starts from 1)
	// if first estimator == 0, include zero predictor (constant)
	// last estimator - the last tree number to predict (enumeration starts from 1)
	Lab_t predictFromTo(const std::vector<FVal_t>& xTest, 
						const size_t firstEstimator, 
						const size_t lastEstimator) const;

	void printModel() const;
protected:
	static std::vector<size_t> sortFeature(const std::vector<FVal_t>& xData);
	void swapAxes(const std::vector<std::vector<FVal_t>>& xTrain);
	static Lab_t loss(const std::vector<Lab_t>& pred, 
					  const std::vector<Lab_t>& truth);
	inline bool canStop(const size_t stepNum, 
						const Lab_t earlyStoppingDelta) const;

	// fields
	size_t featureCount;
	size_t trainLen;
	size_t realTreeCount;
	size_t binCount;
	size_t patience;
	Lab_t zeroPredictor; // constant model
	// xSwapped: 1st dim - feature number, 2nd dim - object number
	std::vector<std::vector<FVal_t>> xSwapped;
	std::vector<GBHist> hists; // histogram for each feature
	std::vector<GBDecisionTree> trees;
	std::vector<Lab_t> trainLosses;
	std::vector<Lab_t> validLosses;

	// constants
	static const size_t defaultBinCount = 128;
	static const float defaultLR;
	static const size_t defaultPatience = 3;
	static constexpr Lab_t defaultESDelta = 0; // for early stopping
};

#endif // GBOOSTING_H

