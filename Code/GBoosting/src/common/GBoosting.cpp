#include "GBoosting.h"
#include "StatisticsHelper.h"

#include <utility>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <cmath>


// static members initialization
const float GradientBoosting::defaultLR = 0.4f;
const unsigned int GradientBoosting::defaultRandomState = 12;

GradientBoosting::GradientBoosting(const size_t binCountMin,
	const size_t binCountMax, const size_t patience,
	const bool dontUseEarlyStopping): featureCount(1), 
	trainLen(0), realTreeCount(0), binCountMin(binCountMin),
	binCountMax(binCountMax), patience(patience), zeroPredictor(0),
	dontUseEarlyStopping(dontUseEarlyStopping) {
	// ctor
	if (binCountMax < binCountMin)
		throw std::runtime_error("Max bin count was less than min bin count");
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
	const size_t treeDepth, const float featureSubsetPart,
	const float learningRate,
	const Lab_t regularizationParam,
	const Lab_t earlyStoppingDelta,
	const float batchPart,
	const bool useJIT,
	const int JITedCodeType,
	const unsigned int randomState,
	const bool shuffledBatches,
	const bool randomThresholds,
	const bool removeRegularizationLater) {
	// Set random seed
	std::srand(randomState);
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
	if (batchPart < 0 || batchPart > 1)
		throw std::runtime_error("batch part was out of [0; 1] interval");
	if (featureSubsetPart <= 0 || featureSubsetPart > 1 )
		throw std::runtime_error("feature fold size was wrong (not in the (0; 1] interval)");
	if (regularizationParam < 0)
		throw std::runtime_error("regularization param was less zero (must be greater or equal)");

	// init tree holder
	// firstly, convert JITed code type to SW_t enum
	SW_t JITedCodeTypeEnum = GradientBoosting::codeTypeToEnum(JITedCodeType);
	// call factory
	treeHolder = TreeHolder::createHolder(useJIT, treeDepth, 
		featureCount, JITedCodeTypeEnum);

	// Histogram init (compute and remember thresholds)
	for (size_t featureSlice = 0; featureSlice < featureCount; ++featureSlice)
		hists.push_back(GBHist(binCountMin, binCountMax, 
			treeCount, xt::col(xTrain, featureSlice), 
			regularizationParam, randomThresholds));
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
	batchSize = size_t(batchPart * trainLen);
	std::vector<size_t> subset = getOrderedIndexes(batchSize);
	// defalt feature subset: all features
	size_t featureSubsetSize = (size_t)round(featureSubsetPart * float(featureCount));
	std::vector<size_t> featureSubset(featureSubsetSize, 0);
	for (size_t i = 0; i < featureSubsetSize; ++i)
		featureSubset[i] = i;
	
	GBDecisionTree::initStaticMembers(learningRate, trainLen, treeDepth);
	GBDecisionTree treeFitter(treeCount, regularizationParam);
	bool stop = false;

	initForRandomBatches(randomState);

	// calculate when remove regularization
	const size_t regularizationKillIter = (size_t)round(float(treeCount) * whenRemoveRegularization);

	for (size_t treeNum = 0; treeNum < treeCount && !stop; ++treeNum) {
		if (removeRegularizationLater && regularizationKillIter == treeNum) {
			// this is the epoch when regularization will be removed
			treeFitter.removeRegularization();
			for (auto & curHist : hists) {
				curHist.removeRegularization();
			}
		}
		
		// take the next batch (updates subset)
		if (shuffledBatches)
			// get the next fold
			nextBatch(subset);
		else
			// get random indexes
			nextBatchRandom(subset);
		// take the next feature subset (updates feature subset)
		nextFeatureSubset(featureSubsetSize, featureCount,
			featureSubset);
		// grow & compile tree
		treeFitter.growTree(xTrain, subset, residuals, hists, featureSubset,
			treeHolder);
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
		if (!dontUseEarlyStopping) {
			stop = canStop(treeNum, earlyStoppingDelta);
			if (stop) {
				break;  // stop fit
			}
		}

		// update historgrams' nets (bin counts)
		for (auto & curHist : hists)
			curHist.updateNet();
	}
	if (!dontUseEarlyStopping && stop) {
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


SW_t GradientBoosting::codeTypeToEnum(const int JITedCodeType) {
	// Firstly, ensure the type number is in [0, MAX_TYPE]
	int enumFit = JITedCodeType % (int(SW_t::SW_COUNT) - 1);
	// Cast int to enum
	return static_cast<SW_t>(JITedCodeType);
}


void GradientBoosting::nextBatch(std::vector<size_t>& allocatedSubset) const {
	// take the next fold
	for (auto & curIdx: allocatedSubset) {
		curIdx = (curIdx + batchSize) % trainLen;
	}
}


void GradientBoosting::nextBatchRandom(std::vector<size_t>& allocatedSubset) {
	// all indexes are splitted into M folds
	// we need to take only one sample from each fold
	// to have subset of size equals batchSize
	// foldCount == batchSize
	
	// shuffle indexes
	std::shuffle(std::begin(shuffledIndexes), 
		std::end(shuffledIndexes), randGenerator);
	
	size_t chosenIdx = 0; // init
	for (size_t fold = 0; fold < batchSize; ++fold) {
		// for each fold get one sample to form a batch
		chosenIdx = randomFromInterval(0, randomFoldLength);
		// get chosenIdx from the current fold from the shuffledIndexes
		allocatedSubset[fold] = shuffledIndexes[fold * randomFoldLength + chosenIdx];
	}
}


void GradientBoosting::nextFeatureSubset(const size_t featureSubsetSize,
	const size_t featureCount,
	std::vector<size_t>& allocatedFeatureSubset) const {
	// take the next fold
	for (auto & curIdx: allocatedFeatureSubset) {
		curIdx = (curIdx + featureSubsetSize) % featureCount;
	}
}


size_t GradientBoosting::randomFromInterval(const size_t left,
		const size_t right) {
	// get random number in [left; right)
	return (right - left) * size_t(rand() / (size_t(RAND_MAX) + 1)) + left;
}


std::vector<size_t> GradientBoosting::getOrderedIndexes(const size_t length) {
	// for length N get std::vector<size_t> = {0, 1, 2, ..., N - 1}
	std::vector<size_t> indexes(length, 0); // init vector
	// fill with indexes
	for (size_t i = 0; i < length; ++i) {
		indexes[i] = i;
	}
	return indexes;
}


void GradientBoosting::initForRandomBatches(const int randomSeed) {
	randGenerator = std::default_random_engine(randomSeed);
	// init array for the indexes that we will shuffle
	shuffledIndexes = getOrderedIndexes(trainLen);
	// all indexes will be splitted into M folds
	// M == batchSize
	// so we will take randomly one index from each fold
	randomFoldLength = size_t(trainLen / batchSize);
}