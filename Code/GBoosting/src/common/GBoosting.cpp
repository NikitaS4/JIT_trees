#include "GBoosting.h"
#include "StatisticsHelper.h"

#include <utility>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <iostream>


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
	if (treeHolder) {
		delete treeHolder;
		treeHolder = nullptr;
	}
}

History GradientBoosting::fit(const pytensor2& xTrain,
	const pytensorY& yTrain, 
	const pytensor2& xValid,
	const pytensorY& yValid, const size_t treeCount,
	const size_t treeDepth, const float learningRate,
	const Lab_t earlyStoppingDelta,
	const bool useJIT) {
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
	
	// init tree holder
	treeHolder = TreeHolder::createHolder(useJIT, treeDepth, featureCount);

	// Histogram init (compute and remember thresholds)
	for (size_t featureSlice = 0; featureSlice < featureCount; ++featureSlice)
		hists.push_back(GBHist(binCount, xt::col(xTrain, featureSlice)));
	// fit ensemble

	// fit the constant model
	zeroPredictor = StatisticsHelper::mean(yTrain);

	// fit another models
	pytensorY residuals = xt::zeros<Lab_t>({trainLen});
	pytensorY preds = xt::zeros<Lab_t>({trainLen});
	pytensorY validRes = xt::zeros<Lab_t>({yValid.shape(0)});
	pytensorY validPreds = xt::zeros<Lab_t>({yValid.shape(0)});
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
		// grow & compile tree
		GBDecisionTree::growTree(xTrain, subset, residuals, hists, treeHolder);
		// update residuals
		for (size_t sample = 0; sample < trainLen; ++sample) {
			Lab_t prediction = treeHolder->predictTree(xt::row(xTrain, sample), 
				treeNum);
			residuals(sample) -= prediction;
			preds(sample) += prediction;
		}
		// update loss
		trainLoss = loss(preds, yTrain);

		// update validation residuals
		for (size_t sample = 0; sample < validLen; ++sample) {
			Lab_t prediction = treeHolder->predictTree(xt::row(xValid, sample),
				treeNum);
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
	}
	if (stop) {
		// need delete the last overfitted estimators
		for (size_t i = 0; i < patience; ++i) {
			treeHolder->popTree();
		}
	}
	realTreeCount = treeHolder->getTreeCount();
	return History(realTreeCount, trainLosses, validLosses);
}

Lab_t GradientBoosting::predict(const pytensor1& xTest) const {
	if (xTest.shape(0) != featureCount)
		throw std::runtime_error("Wrong feature count in x_test");
	return zeroPredictor + treeHolder->predictAllTrees(xTest);
}

pytensorY GradientBoosting::predict(const pytensor2& xTest) const {
	size_t predCount = xTest.shape(0);
	size_t features = xTest.shape(1);
	if (features != featureCount)
		throw std::runtime_error("Wrong feature count in x_test"); 
	pytensorY preds = xt::zeros<Lab_t>({predCount});
	for (size_t i = 0; i < predCount; ++i) {		
		preds(i) = zeroPredictor + treeHolder->predictAllTrees(xt::row(xTest, i));
	}
	return preds;
}


Lab_t GradientBoosting::predictFromTo(const pytensor1& xTest, 
	const size_t firstEstimator, const size_t lastEstimator) const {
	Lab_t curPred = 0;
	size_t from = firstEstimator;
	if (xTest.shape(0) != featureCount)
		throw std::runtime_error("Wrong feature count in x_test");
	if (firstEstimator > lastEstimator)
		throw std::runtime_error("Order of the first and the last estimators is inverted");
	if (lastEstimator > realTreeCount)
		throw std::runtime_error("Too big last estimator, ensemble contain less trees number");
	if (firstEstimator == 0) { // use zero estimator (constant)
		curPred = zeroPredictor;
		++from;
	}
	return curPred + treeHolder->predictFromTo(xTest, from - 1, lastEstimator - 1);	
}


Lab_t GradientBoosting::loss(const pytensorY& pred,
	const pytensorY& truth) {
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
