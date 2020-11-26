#include "GBoosting.h"
#include "StatisticsHelper.h"

#include <utility>
#include <iostream>
#include <algorithm>
#include <functional>


// static members initialization
const float GradientBoosting::defaultLR = 0.4f;

GradientBoosting::GradientBoosting(const size_t binCount,
	const size_t patience): featureCount(1), 
	trainLen(0), realTreeCount(0), binCount(binCount), 
	patience(patience), zeroPredictor(0) {
	// ctor
}

GradientBoosting::~GradientBoosting() {
	// dtor
}

History GradientBoosting::fit(const std::vector<std::vector<FVal_t>>& xTrain,
	const std::vector<Lab_t>& yTrain, 
	const std::vector<std::vector<FVal_t>>& xValid,
	const std::vector<Lab_t>& yValid, const size_t treeCount,
	const size_t treeDepth, const float learningRate) {
	// Prepare data	
	trainLen = xTrain.size();
	featureCount = xTrain[0].size();
	swapAxes(xTrain);  // x.shape = (featureCount, trainLen)
	// Now it's easy to pass feature slices to build histogram
	// Histogram building
	for (auto& featureSlice : xSwapped) {
		std::vector<size_t> backIdxs;
		std::vector<size_t> sortedIdxs = sortFeature(featureSlice, backIdxs);
		hists.push_back(GBHist(binCount, sortedIdxs, backIdxs, featureSlice));
	}
	// fit ensemble

	// fit the constant model
	zeroPredictor = StatisticsHelper::mean(yTrain);

	// fit another models
	std::vector<Lab_t> residuals;
	std::vector<Lab_t> preds;
	std::vector<Lab_t> validRes;
	std::vector<Lab_t> validPreds;
	Lab_t trainLoss;
	Lab_t validLoss;
	// residuals = yTest - trainPreds
	for (size_t i = 0; i < trainLen; ++i) {
		residuals.push_back(yTrain[i] - zeroPredictor);
		preds.push_back(zeroPredictor);
	}

	trainLoss = loss(preds, yTrain);  // update loss

	// validation residuals
	size_t validLen = yValid.size();
	for (size_t i = 0; i < validLen; ++i) {
		validRes.push_back(yValid[i] - zeroPredictor);
		validPreds.push_back(zeroPredictor);
	}
	validLoss = loss(validPreds, yValid);  // update loss
	
	// remember losses
	trainLosses.push_back(trainLoss);
	validLosses.push_back(validLoss);

	// default subset: all data
	std::vector<size_t> subset;
	for (size_t i = 0; i < trainLen; ++i)
		subset.push_back(i);
	GBDecisionTree::initStaticMembers(learningRate, treeDepth);
	bool stop = false;

	for (size_t treeNum = 0; treeNum < treeCount && !stop; ++treeNum) {
		GBDecisionTree curTree(xSwapped, subset, residuals, hists);
		// update residuals
		for (size_t sample = 0; sample < trainLen; ++sample) {
			Lab_t prediction = curTree.predict(xTrain[sample]);
			residuals[sample] -= prediction;
			preds[sample] += prediction;
		}
		// update loss
		trainLoss = loss(preds, yTrain);

		// update validation residuals
		for (size_t sample = 0; sample < validLen; ++sample) {
			Lab_t prediction = curTree.predict(xValid[sample]);
			validRes[sample] -= prediction;
			validPreds[sample] += prediction;
		}
		// update validation loss
		validLoss = loss(validPreds, yValid);
		
		// remember losses
		trainLosses.push_back(trainLoss);
		validLosses.push_back(validLoss);

		// update losses difference
		stop = canStop(treeNum);
		if (stop) {
			break;  // stop fit
		}
		trees.emplace_back(std::move(curTree));
	}
	if (stop) {
		// need delete the last overfitted estimators
		for (size_t i = 0; i < patience; ++i) {
			trees.pop_back();
		}
	}
	realTreeCount = trees.size();  // without early stopping
	return History(realTreeCount, trainLosses, validLosses);
}

Lab_t GradientBoosting::predict(const std::vector<FVal_t>& xTest) const {
	Lab_t curPred = zeroPredictor;
	for (auto& curTree : trees)
		curPred += curTree.predict(xTest);
	return curPred;
}


void GradientBoosting::printModel() const {
	std::cout << "Printing Gradient boosting model\n";

	std::cout << "Zero predictor: " << zeroPredictor << "\n";

	for (size_t tree = 0; tree < realTreeCount; ++tree) {
		std::cout << "Tree " << tree << "\n";
		trees[tree].printTree();
	}

	std::cout << "====================\n";
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


std::vector<size_t> GradientBoosting::sortFeature(const std::vector<FVal_t>& xData,
	std::vector<size_t>& backIdxs) {
	// wierd bubble sort, the first implementation

	size_t n = xData.size();  // data len
	std::vector<size_t> sortedIdxs;

	for (size_t i = 0; i < n; ++i) {
		sortedIdxs.push_back(i);  // at once, order as in xData
	}

	auto comparator = [&xData](size_t a, size_t b)->bool {
		return xData[a] < xData[b];
	};
	std::sort(sortedIdxs.begin(), sortedIdxs.end(), comparator);

	backIdxs = std::vector<size_t>(sortedIdxs.size(), 0);
	for (size_t i = 0; i < sortedIdxs.size(); ++i) {
		backIdxs[sortedIdxs[i]] = i;
	}

	return sortedIdxs;
}

Lab_t GradientBoosting::loss(const std::vector<Lab_t>& pred,
	const std::vector<Lab_t>& truth) {
	// 0.5 * MSE
	size_t count = pred.size();
	Lab_t squaredErrorSum = 0;
	Lab_t res = 0;
	for (size_t i = 0; i < count; ++i) {
		res = pred[i] - truth[i];
		squaredErrorSum += (res * res) / 2;
	}
	return squaredErrorSum / count;
}

bool GradientBoosting::canStop(const size_t stepNum) const {
	if (stepNum < patience) {
		return false;
	}
	else {
		for (size_t i = stepNum - patience + 1; i <= stepNum; ++i) {
			if (validLosses[i] < validLosses[i - 1])
				return false;
		}
		return true;
	}
}
