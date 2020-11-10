#include "GBoosting.h"
#include "StatisticsHelper.h"


GradientBoosting::GradientBoosting() {
	// ctor
}

GradientBoosting::~GradientBoosting() {
	// dtor
}

void GradientBoosting::fit(const std::vector<std::vector<FVal_t>>& xTest,
	const std::vector<Lab_t>& yTest) {

	// Prepare data	
	trainLen = xTest.size();
	featureCount = xTest[0].size();
	swapAxes(xTest);  // x.shape = (featureCount, trainLen)
	// Now it's easy to pass feature slices to build histogram

	// Histogram building
	for (auto& featureSlice : xSwapped) {
		std::vector<size_t> sortedIdxs;
		sortFeature(featureSlice, sortedIdxs);
		size_t binCount = 4;  // TODO: make static class constant or parameter or both
		hists.push_back(GBHist(binCount, sortedIdxs, featureSlice));
	}

	// fit ensemble

	// fit the constant model
	zeroPredictor = StatisticsHelper::mean(yTest);

	// fit another models
	treeCount = 3;  // TODO: make a parameter
	std::vector<Lab_t> residuals(yTest);  // deep copy - rewrite later
	// residuals = yTest - trainPreds

	// default subset: all data
	std::vector<size_t> subset;
	for (size_t i = 0; i < trainLen; ++i)
		subset.push_back(i);

	for (size_t treeNum = 0; treeNum < treeCount; ++treeNum) {
		// TODO: add early stopping
		GBDecisionTree curTree(xSwapped, subset, residuals, hists);
		trees.push_back(curTree);
		// update residuals
		for (size_t sample = 0; sample < trainLen; ++sample)
			residuals[sample] -= curTree.predict(xTest[sample]);
	}
}

Lab_t GradientBoosting::predict(const std::vector<FVal_t>& xTest) {
	Lab_t curPred = zeroPredictor;
	for (auto& curTree : trees)
		curPred += curTree.predict(xTest);
	return curPred;
}


void GradientBoosting::swapAxes(const std::vector<std::vector<FVal_t>>& xTest) {
	std::vector<FVal_t> featureVals;
	for (size_t feature = 0; feature < featureCount; ++featureCount) {
		featureVals.clear();
		for (size_t dataNum = 0; dataNum < trainLen; ++dataNum) {
			featureVals.push_back(xTest[dataNum][feature]);
		}
		xSwapped.push_back(featureVals);
	}
}


void GradientBoosting::sortFeature(const std::vector<FVal_t>& xData,
	std::vector<size_t>& sortedIdxs) {
	// wierd bubble sort, the first implementation

	size_t n = xData.size();  // data len
	sortedIdxs.clear();  // clear array

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
}