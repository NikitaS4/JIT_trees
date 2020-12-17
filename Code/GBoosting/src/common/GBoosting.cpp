#include "GBoosting.h"
#include "StatisticsHelper.h"

#include <utility>
#include <stdexcept>
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

History GradientBoosting::fit(const pyarray& xTrain,
	const pyarrayY& yTrain, 
	const pyarray& xValid,
	const pyarrayY& yValid, const size_t treeCount,
	const size_t treeDepth, const float learningRate,
	const Lab_t earlyStoppingDelta) {
	// Prepare data	
	trainLen = xTrain.shape(0);
	featureCount = xTrain.shape(1);

	if (xTrain.shape().size() != 2)
		throw std::runtime_error("xTrain - wrong shape");
	if (yTrain.shape().size() != 1)
		throw std::runtime_error("yTrain - wrong shape");
	if (xValid.shape().size() != 2)
		throw std::runtime_error("xValid - wrong shape");
	if (yValid.shape().size() != 1)
		throw std::runtime_error("yValid - wrong shape");
	if (yTrain.shape(0) != trainLen)
		throw std::runtime_error("xTrain & yTrain sizes mismatch");
	if (xValid.shape(0) != yValid.shape(0))
		throw std::runtime_error("xValid & yValid sizes mismatch");
	if (xValid.shape(1) != featureCount)
		throw std::runtime_error("xValid feature dimension wrong");
	if (earlyStoppingDelta < 0)
		throw std::runtime_error("early stopping delta was negative");
	
	// Histogram building
	for (size_t featureSlice = 0; featureSlice < xTrain.shape(1); ++featureSlice) {
		std::vector<size_t> sortedIdxs = sortFeature(xt::col(xTrain, featureSlice));
		hists.push_back(GBHist(binCount, sortedIdxs, xt::col(xTrain, featureSlice)));
	}
	// fit ensemble

	// fit the constant model
	zeroPredictor = StatisticsHelper::mean(yTrain);

	// fit another models
	pyarrayY residuals = xt::zeros<Lab_t>({trainLen});
	pyarrayY preds = xt::zeros<Lab_t>({trainLen});
	pyarrayY validRes = xt::zeros<Lab_t>({yValid.shape(0)});
	pyarrayY validPreds = xt::zeros<Lab_t>({yValid.shape(0)});
	Lab_t trainLoss;
	Lab_t validLoss;
	// residuals = yTest - trainPreds
	for (size_t i = 0; i < trainLen; ++i) {
		residuals(i) = yTrain(i) - zeroPredictor;
		preds(i) = zeroPredictor;
	}

	trainLoss = loss(preds, yTrain);  // update loss

	// validation residuals
	size_t validLen = yValid.size();
	for (size_t i = 0; i < validLen; ++i) {
		validRes(i) = yValid(i) - zeroPredictor;
		validPreds(i) = zeroPredictor;
	}
	validLoss = loss(validPreds, yValid);  // update loss
	
	// remember losses
	// treeCount + 1 -- to include zero predictor
	trainLosses = xt::zeros<Lab_t>({treeCount + 1});
	validLosses = xt::zeros<Lab_t>({treeCount + 1});
	trainLosses(0) = trainLoss;
	validLosses(0) = validLoss;

	// default subset: all data
	std::vector<size_t> subset;
	for (size_t i = 0; i < trainLen; ++i)
		subset.push_back(i);
	GBDecisionTree::initStaticMembers(learningRate, trainLen, treeDepth);
	bool stop = false;

	for (size_t treeNum = 0; treeNum < treeCount && !stop; ++treeNum) {
		GBDecisionTree curTree(xTrain, subset, residuals, hists);
		// update residuals
		for (size_t sample = 0; sample < trainLen; ++sample) {
			Lab_t prediction = curTree.predict(xTrain(sample));
			residuals(sample) -= prediction;
			preds(sample) += prediction;
		}
		// update loss
		trainLoss = loss(preds, yTrain);

		// update validation residuals
		for (size_t sample = 0; sample < validLen; ++sample) {
			Lab_t prediction = curTree.predict(xValid(sample));
			validRes(sample) -= prediction;
			validPreds(sample) += prediction;
		}
		// update validation loss
		validLoss = loss(validPreds, yValid);
		
		// remember losses
		trainLosses(treeNum + 1) = trainLoss;
		validLosses(treeNum + 1) = validLoss;

		// update losses difference
		stop = canStop(treeNum, earlyStoppingDelta);
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

Lab_t GradientBoosting::predict(const pyarray& xTest) const {
	Lab_t curPred = zeroPredictor;
	for (auto& curTree : trees)
		curPred += curTree.predict(xTest);
	return curPred;
}


Lab_t GradientBoosting::predictFromTo(const pyarray& xTest, 
	const size_t firstEstimator, const size_t lastEstimator) const {
	Lab_t curPred = 0;
	if (firstEstimator > lastEstimator)
		throw std::runtime_error("Order of the first and the last estimators is inverted");
	if (lastEstimator > realTreeCount)
		throw std::runtime_error("Too big last estimator, ensemble contain less trees number");
	if (firstEstimator == 0) { // use zero estimator (constant)
		curPred = zeroPredictor;
	}
	for (size_t estimatorNum = firstEstimator; estimatorNum <= lastEstimator; ++estimatorNum) {
		if (estimatorNum == 0)
			continue; // zero estimator already used
		curPred += trees[estimatorNum - 1].predict(xTest);
	}
	return curPred;
}


std::vector<size_t> GradientBoosting::sortFeature(const pyarray& xData) {
	size_t n = xData.shape(0);  // data len
	std::vector<size_t> sortedIdxs;

	for (size_t i = 0; i < n; ++i) {
		sortedIdxs.push_back(i);  // at once, order as in xData
	}

	auto comparator = [&xData](size_t a, size_t b)->bool {
		return xData(a) < xData(b);
	};
	std::sort(sortedIdxs.begin(), sortedIdxs.end(), comparator);

	return sortedIdxs;
}

Lab_t GradientBoosting::loss(const pyarrayY& pred,
	const pyarrayY& truth) {
	// 0.5 * MSE
	size_t count = pred.size();
	Lab_t squaredErrorSum = 0;
	Lab_t res = 0;
	for (size_t i = 0; i < count; ++i) {
		res = pred(i) - truth(i);
		squaredErrorSum += (res * res) / 2;
	}
	return squaredErrorSum / count;
}

bool GradientBoosting::canStop(const size_t stepNum, 
	const Lab_t earlyStoppingDelta) const {
	if (stepNum < patience) {
		return false;
	}
	else {
		for (size_t i = stepNum - patience + 1; i <= stepNum; ++i) {
			if (validLosses(i) - validLosses(i - 1) < -earlyStoppingDelta)
				return false;
		}
		return true;
	}
}
