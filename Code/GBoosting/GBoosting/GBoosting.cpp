#include "GBoosting.h"
#include "StatisticsHelper.h"

#include <utility>


GradientBoosting::GradientBoosting(const size_t binCount): featureCount(1), 
	trainLen(0), realTreeCount(0), binCount(binCount), 
	zeroPredictor(0) {
	// ctor
}

GradientBoosting::~GradientBoosting() {
	// dtor
}

void GradientBoosting::fit(const std::vector<std::vector<FVal_t>>& xTest,
	const std::vector<Lab_t>& yTest, const size_t treeCount,
	const size_t treeDepth) {
	// Prepare data	
	trainLen = xTest.size();
	featureCount = xTest[0].size();
	swapAxes(xTest);  // x.shape = (featureCount, trainLen)
	// Now it's easy to pass feature slices to build histogram
	// Histogram building
	for (auto& featureSlice : xSwapped) {
		std::vector<size_t> sortedIdxs = sortFeature(featureSlice);
		hists.push_back(GBHist(binCount, sortedIdxs, featureSlice));
	}

	// fit ensemble

	// fit the constant model
	zeroPredictor = StatisticsHelper::mean(yTest);

	// fit another models
	std::vector<Lab_t> residuals;
	// residuals = yTest - trainPreds
	for (size_t i = 0; i < trainLen; ++i)
		residuals.push_back(yTest[i] - zeroPredictor);

	// default subset: all data
	std::vector<size_t> subset;
	for (size_t i = 0; i < trainLen; ++i)
		subset.push_back(i);
	GBDecisionTree::initTreeDepth(treeDepth);
	for (size_t treeNum = 0; treeNum < treeCount; ++treeNum) {
		// TODO: add early stopping
		GBDecisionTree curTree(xSwapped, subset, residuals, hists);
		// update residuals
		for (size_t sample = 0; sample < trainLen; ++sample)
			residuals[sample] -= curTree.predict(xTest[sample]);
		trees.emplace_back(std::move(curTree));
	}
	realTreeCount = treeCount;  // without early stopping
}

Lab_t GradientBoosting::predict(const std::vector<FVal_t>& xTest) const {
	Lab_t curPred = zeroPredictor;
	for (auto& curTree : trees)
		curPred += curTree.predict(xTest);
	return curPred;
}


void GradientBoosting::swapAxes(const std::vector<std::vector<FVal_t>>& xTest) {
	std::vector<FVal_t> featureVals;
	for (size_t feature = 0; feature < featureCount; ++feature) {
		featureVals.clear();
		for (size_t dataNum = 0; dataNum < trainLen; ++dataNum) {
			featureVals.push_back(xTest[dataNum][feature]);
		}
		xSwapped.push_back(featureVals);
	}
}


std::vector<size_t> GradientBoosting::sortFeature(const std::vector<FVal_t>& xData) {
	// wierd bubble sort, the first implementation

	size_t n = xData.size();  // data len
	std::vector<size_t> sortedIdxs;

	for (size_t i = 0; i < n; ++i) {
		sortedIdxs.push_back(i);  // at once, order as in xData
	}

	for (size_t i = 0; i < n - 1; ++i) {
		for (size_t j = 0; j < n - i - 1; ++j) {
			if (xData[sortedIdxs[j]] > xData[sortedIdxs[j + 1]]) {
				// invertion has to be performed
				std::swap(sortedIdxs[j + 1], sortedIdxs[j]);
			}
		}
	}
	return sortedIdxs;
}