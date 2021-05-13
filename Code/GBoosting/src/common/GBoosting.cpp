#include "GBoosting.h"
#include "StatisticsHelper.h"
#include "ParseHelper.h"

#include <utility>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <cstdio>


GradientBoosting::GradientBoosting(const size_t binCountMin,
	const size_t binCountMax, const size_t patience,
	const bool dontUseEarlyStopping,
	const size_t threadCnt): featureCount(1), 
	trainLen(0), realTreeCount(0), binCountMin(binCountMin),
	binCountMax(binCountMax), patience(patience), threadCnt(threadCnt),
	zeroPredictor(0), dontUseEarlyStopping(dontUseEarlyStopping) {
	// ctor
	if (binCountMax < binCountMin)
		throw std::runtime_error("Max bin count was less than min bin count");
	if (threadCnt == 0)
		throw std::runtime_error("Thread count was 0 (must be positive)");
}

GradientBoosting::~GradientBoosting() {
	// dtor
	if (predictor) {
		delete predictor;
		predictor = nullptr;
	}

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
	const unsigned int randomState,
	const bool randomBatches,
	const bool randomThresholds,
	const bool removeRegularizationLater,
	const bool spoilScores) {
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
	// call factory
	treeHolder = TreeHolder::createHolder(treeDepth, featureCount,
		threadCnt);

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
	
	// create predictor
	predictor = GBPredictor::create(zeroPredictor, *treeHolder,
		&xTrain, &xValid, &residuals, &preds, &validRes, &validPreds);
	if (predictor == nullptr)
		throw std::runtime_error("Can't fit: not enough memory");

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
	GBDecisionTree treeFitter(treeCount, regularizationParam,
		spoilScores);
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
		if (randomBatches)
			// get the next fold
			nextBatch(subset);
		else
			// get random indexes
			nextBatchRandom(subset);
		// take the next feature subset (updates feature subset)
		nextFeatureSubset(featureSubsetSize, featureCount,
			featureSubset);
		// grow & compile tree
		treeFitter.growTree(xTrain, subset, residuals, featureSubset,
			hists, treeHolder);
		// update residuals
		predictor->predictTreeTrain(treeNum);
		
		// update losses
		trainLoss = loss(preds, yTrain);
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
	return predictor->predict1d(xTest);
}

pytensorY GradientBoosting::predict(const pytensor2& xTest) const {
	return predictor->predict2d(xTest);
}


Lab_t GradientBoosting::predictFromTo(const pytensor1& xTest, 
	// TODO: use predictor instead
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


void GradientBoosting::saveModel(const std::string& fname) const {
	// Save file structure:
	// <Type><d><FeatureCnt><d><TreeCount><d><TreeDepth><d><zeroPredictor><d><Trees><e>
	// <Type> ::= 0 | 1  # 0 for classification, 1 for regression
	// <Trees> ::= <Tree> | <Tree><d><Trees>
	// <Tree> ::= <Features><d><Thresholds><d><Leaves>
	// <Features> ::= <Feature> | <Feature><d><Features>
	// <Thresholds> ::= <Threshold> | <Threshold><d><Thresholds>
	// <Leaves> ::= <Leaf> | <Leaf><d><Leaves>
	// <Feature> ::= <size_t number>
	// <Threshold> ::= <FVal_t number>
	// <Leaf> ::= <Lab_t number>
	// <d> ::= ;  # delimeter
	// <e> ::= !  # end
	
	std::string contents;
	static const char delimeter = ';';
	static const char modelEnd = '!';
	// try open file to write
	// if file does not exist, an exception will be thrown
	std::ofstream outfile(fname);
	// <Type><d>
	contents += std::to_string(modelType) + delimeter;
	// <FeatureCnt><d>
	contents += std::to_string(featureCount) + delimeter;
	// <TreeCount><d><TreeDepth><d><zeroPredictor><d><Trees>
	contents += treeHolder->serialize(delimeter, zeroPredictor);
	// <e>
	contents += modelEnd;
	// now contents are created properly
	// write the contents to the file with a single operation
	outfile << contents;
	outfile.close();
}


GradientBoosting::GradientBoosting(const std::string& fname,
	const size_t threadCnt): threadCnt(threadCnt) {	
	// File structure:
	// <Type><d><FeatureCnt><d><TreeCount><d><TreeDepth><d><zeroPredictor><d><Trees><e>
	// <Type> ::= 0 | 1  # 0 for classification, 1 for regression
	// <Trees> ::= <Tree> | <Tree><d><Trees>
	// <Tree> ::= <Features><d><Thresholds><d><Leaves>
	// <Features> ::= <Feature> | <Feature><d><Features>
	// <Thresholds> ::= <Threshold> | <Threshold><d><Thresholds>
	// <Leaves> ::= <Leaf> | <Leaf><d><Leaves>
	// <Feature> ::= <size_t number>
	// <Threshold> ::= <FVal_t number>
	// <Leaf> ::= <Lab_t number>
	// <d> ::= ;  # delimeter
	// <e> ::= !  # end
	static const char delimeter = ';';
	static const char modelEnd = '!';
	FILE* pFile = fopen(fname.c_str(), "r");
	if (pFile == nullptr) {
		throw std::runtime_error("Can't open the file");
	}
	// get file size
	size_t fileSize = 0;
	fseek(pFile, 0, SEEK_END);
	fileSize = ftell(pFile);
	// read the file with a single operation
	fseek(pFile, 0, SEEK_SET);
	char* contents = (char*)malloc(fileSize * sizeof(char));
	if (contents == nullptr) {
		fclose(pFile);
		throw std::runtime_error("Not enough memory to read the file");
	}
	size_t symsRead = fread(contents, sizeof(char), fileSize, pFile);
	// close the file
	fclose(pFile);
	if (symsRead != fileSize) {
		free(contents);
		throw std::runtime_error("Error occurred while reading the file");
	}
	// parse the contents
	// step 1: find all delimeter occurrences

    std::vector<size_t> delimPositions;
    for (size_t i = 0; i < fileSize; ++i) {
        if (contents[i] == delimeter) {
            delimPositions.push_back(i);
        }
        if (contents[i] == modelEnd) {
            delimPositions.push_back(i);
            break;
        }
    }
	static const size_t dPosMinSize = 5; // at least only zero predictor
	if (delimPositions.size() < dPosMinSize) {
		free(contents);
		throw std::runtime_error("Can't load model: invalid file");
	}

	// step 2: parse initial info for the tree holder
	// we will take each time the next symbol to the delimeter
    const char* nextSym = contents + 1;
    size_t curDelimeterIdx = 0; // skip <Type>
	featureCount = ParseHelper::parseSizeT(nextSym + delimPositions[curDelimeterIdx++]);
    realTreeCount = ParseHelper::parseSizeT(nextSym + delimPositions[curDelimeterIdx++]);
    size_t treeDepth = ParseHelper::parseSizeT(nextSym + delimPositions[curDelimeterIdx++]);	 
	if (!valCptContents(delimPositions, modelEnd, contents,
		realTreeCount, treeDepth, dPosMinSize)) {
		free(contents);
		throw std::runtime_error("Can't load model: invalid file");
	}
    zeroPredictor = (Lab_t)ParseHelper::parseFloat(nextSym + delimPositions[curDelimeterIdx++]);

	treeHolder = TreeHolder::parse(nextSym, delimPositions, curDelimeterIdx,
		featureCount, realTreeCount, treeDepth, threadCnt);
	
	free(contents);
	// check result
	if (treeHolder == nullptr)
		throw std::runtime_error("Not enough memory to load model");
	predictor = GBPredictor::createReady(zeroPredictor, *treeHolder,
		featureCount);
	if (predictor == nullptr) {
		delete treeHolder;
		treeHolder = nullptr;
		throw std::runtime_error("Not enough memory to load model");
	}
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


bool GradientBoosting::valCptContents(const std::vector<size_t>& dPos,
		const char modelEnd, char const * const contents,
		const size_t treeCnt, const size_t treeDepth,
		const size_t dPosMinSize) const {
	size_t dPosSize = dPos.size();
	// check minimum size
	if (dPosMinSize > dPosSize)
		return false;
	// check feature count
	if (featureCount == 0)
		return false;
	// check the last delimeter equals model end
	if (contents[dPos[dPosSize - 1]] != modelEnd)
		return false;
	// check true size
	size_t innerNodes = (size_t(1) << treeDepth) - 1;
	size_t leafCnt = size_t(1) << treeDepth;
	size_t rightSize = (treeDepth + innerNodes + leafCnt) * treeCnt + dPosMinSize;
	if (dPosSize != rightSize)
		return false;
	// check substrings between delimeters
	for (size_t i = 1; i < dPosSize; ++i) {
		if (dPos[i] - dPos[i - 1] <= 1)
			return false;
	}
	// all checks passed
	return true;
}
