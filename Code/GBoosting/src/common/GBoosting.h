#ifndef GBOOSTING_H
#define GBOOSTING_H

#include "PybindHeader.h"
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
	History fit(const pytensor2& xTrain, 
				const pytensorY& yTrain,
				const pytensor2& xValid,
				const pytensorY& yValid,
				const size_t treeCount,
				const size_t treeDepth,
				const float learningRate = defaultLR,
				const Lab_t earlyStoppingDelta = defaultESDelta);
	Lab_t predict(const pytensor1& xTest) const;
	pytensorY predict(const pytensor2& xTest) const;

	// predict "from-to" - predict using only subset of trees
	// first estimator - the first tree number to predict (enumeration starts from 1)
	// if first estimator == 0, include zero predictor (constant)
	// last estimator - the last tree number to predict (enumeration starts from 1)
	Lab_t predictFromTo(const pytensor1& xTest, 
						const size_t firstEstimator, 
						const size_t lastEstimator) const;

protected:
	static Lab_t loss(const pytensorY& pred, 
					  const pytensorY& truth);
	inline bool canStop(const size_t stepNum, 
						const Lab_t earlyStoppingDelta) const;

	// fields
	size_t featureCount;
	size_t trainLen;
	size_t realTreeCount;
	size_t binCount;
	size_t patience;
	Lab_t zeroPredictor; // constant model
	std::vector<GBHist> hists; // histogram for each feature
	std::vector<GBDecisionTree> trees;
	pytensorY trainLosses;
	pytensorY validLosses;

	// constants
	static const size_t defaultBinCount = 128;
	static const float defaultLR;
	static const size_t defaultPatience = 3;
	static constexpr Lab_t defaultESDelta = 0; // for early stopping
};

#endif // GBOOSTING_H

