#ifndef GBOOSTING_H
#define GBOOSTING_H

#include "PybindHeader.h"
#include "Structs.h"
#include "GBHist.h"
#include "GBDecisionTree.h"
#include "History.h"
#include "../TreeHolders/TreeHolder.h"
#include "GBPredictor.h"
#include <vector>
#include <random>
#include <string>


class GradientBoosting {
public:
	GradientBoosting(const size_t binCountMin,
					 const size_t binCountMax,
					 const size_t patience,
					 const bool dontUseEarlyStopping,
					 const size_t threadCnt);
	virtual ~GradientBoosting();
	// 1st dim - object number, 2nd dim - feature number
	// fit return the number of estimators (include constant estim)
	History fit(const pytensor2& xTrain, 
				const pytensorY& yTrain,
				const pytensor2& xValid,
				const pytensorY& yValid,
				const size_t treeCount,
				const size_t treeDepth,
				const float featureSubsetPart,
				const float learningRate,
				const Lab_t regularizationParam,
				const Lab_t earlyStoppingDelta,
				const float batchPart,
				const unsigned int randomState,
				const bool shuffledBatches,
				const bool randomThresholds,
				const bool removeRegularizationLater);
	Lab_t predict(const pytensor1& xTest) const;
	pytensorY predict(const pytensor2& xTest) const;

	// predict "from-to" - predict using only subset of trees
	// first estimator - the first tree number to predict (enumeration starts from 1)
	// if first estimator == 0, include zero predictor (constant)
	// last estimator - the last tree number to predict (enumeration starts from 1)
	Lab_t predictFromTo(const pytensor1& xTest, 
						const size_t firstEstimator, 
						const size_t lastEstimator) const;

	void saveModel(const std::string& fname) const;
	void loadModel(const std::string& fname);

protected:
	static Lab_t loss(const pytensorY& pred, 
					  const pytensorY& truth);
	inline bool canStop(const size_t stepNum, 
						const Lab_t earlyStoppingDelta) const;
 
	static inline size_t randomFromInterval(const size_t left,
		const size_t right);

	static inline std::vector<size_t> getOrderedIndexes(const size_t length);

	inline void nextBatch(std::vector<size_t>& allocatedSubset) const;

	inline void nextBatchRandom(std::vector<size_t>& allocatedSubset);

	inline void nextFeatureSubset(const size_t featureSubsetSize,
		const size_t featureCount,
		std::vector<size_t>& allocatedFeatureSubset) const;

	inline void initForRandomBatches(const int randomSeed);
	
	inline bool valCptContents(const std::vector<size_t>& dPos,
		const char modelEnd, char const * const contents,
		const size_t treeCnt, const size_t treeDepth,
		const size_t dPosMinSize) const;

	// fields
	size_t featureCount;
	size_t trainLen;
	size_t realTreeCount;
	size_t binCountMin;
	size_t binCountMax;
	size_t patience;
	size_t randomFoldLength; // it's needed to form random batches
	const size_t threadCnt;
	std::vector<size_t> shuffledIndexes; // it's needed to form random batches
	std::default_random_engine randGenerator;
	size_t batchSize;
	Lab_t zeroPredictor; // constant model
	std::vector<GBHist> hists; // histogram for each feature
	pytensorY trainLosses;
	pytensorY validLosses;
	bool dontUseEarlyStopping; // switch off early stopping
	TreeHolder* treeHolder = nullptr;
	GBPredictor* predictor = nullptr;

	// constants
	static constexpr float whenRemoveRegularization = 0.8f; // the part of iterations with regularization	
	static constexpr size_t modelType = 1; // regression (0 for classification)
};

#endif // GBOOSTING_H

