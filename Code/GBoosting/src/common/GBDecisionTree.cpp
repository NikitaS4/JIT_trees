#include "GBDecisionTree.h"
#include "StatisticsHelper.h"

#include <stdexcept>


// static initializations
bool GBDecisionTree::depthAssigned = false;
size_t GBDecisionTree::featureCount = 0;
size_t GBDecisionTree::treeDepth = 0;
size_t GBDecisionTree::innerNodes = 0;
size_t GBDecisionTree::leafCnt = 0;
float GBDecisionTree::learningRate = 1;
std::vector<std::vector<size_t>> GBDecisionTree::subset;


void GBDecisionTree::initStaticMembers(const float learnRate,
	const size_t trainLen, const size_t depth) {
	if (depth == 0)
		throw std::runtime_error("Wrong tree depth");
	treeDepth = depth;
	innerNodes = (1 << treeDepth) - 1;
	leafCnt = size_t(1) << treeDepth;
	subset = std::vector<std::vector<size_t>>(innerNodes + leafCnt,
		std::vector<size_t>());
	depthAssigned = true;
	learningRate = learnRate;
}

void GBDecisionTree::growTree(const pytensor2& xTrain,
	const std::vector<size_t>& chosen, 
	const pytensorY& yTrain,
	const std::vector<GBHist>& hists,
	TreeHolder* treeHolder) {
	if (!depthAssigned)
		throw std::runtime_error("Tree depth was not assigned");
	size_t* features = new size_t[treeDepth];
	FVal_t* thresholds = new FVal_t[innerNodes];
	for (size_t i = 0; i < innerNodes; ++i)
		thresholds[i] = 0;
	Lab_t* leaves = new Lab_t[leafCnt];
	for (size_t i = 0; i < leafCnt; ++i)
		leaves[i] = 0;
	pytensor2Y intermediateLabels = xt::zeros<Lab_t>({innerNodes + leafCnt, xTrain.shape(0)});

	// 1st dim - node number, 2nd dim - sample idx
	subset[0] = chosen;	
	xt::row(intermediateLabels, 0) = yTrain;

	featureCount = xTrain.shape(1);

	size_t broCount = 1;
	std::vector<FVal_t> bestThreshold;
	Lab_t bestScore;
	bool firstSplitFound = false;
	Lab_t curScore;
	std::vector<FVal_t> curThreshold(leafCnt, 0);
	FVal_t atomicThreshold;
	size_t bestFeature = 0;
	for (size_t h = 0; h < treeDepth; ++h) {
		// find best split
		size_t firstBroNum = (1 << h) - 1;
		for (size_t feature = 0; feature < featureCount; ++feature) {
			// for all nodes look for the best split of the feature

			curScore = 0;
			for (size_t node = 0; node < broCount; ++node) {
				// find best score
				curScore += hists[feature].findBestSplit(xt::col(xTrain, feature),
					subset[firstBroNum + node], 
					xt::row(intermediateLabels, firstBroNum + node), atomicThreshold);
				curThreshold[node] = atomicThreshold;
			}
			if (!firstSplitFound || curScore < bestScore) {
				bestScore = curScore;
				bestThreshold = curThreshold;
				bestFeature = feature;
			}
		}
		// the best score is found now
		// need to perform the split
		for (size_t node = 0; node < broCount; ++node) {
			std::vector<size_t> leftSubset;
			std::vector<size_t> rightSubset;
			size_t absoluteNode = firstBroNum + node;
			thresholds[absoluteNode] = bestThreshold[node];
			leftSubset = hists[bestFeature].performSplit(xt::col(xTrain, bestFeature),
				subset[absoluteNode], bestThreshold[node], rightSubset);
			// subset will be placed to their topological places
			size_t leftSon = 2 * absoluteNode + 1;
			size_t rightSon = leftSon + 1;
			subset[leftSon] = leftSubset;
			subset[rightSon] = rightSubset;
			// update labels
			Lab_t leftAvg = StatisticsHelper::mean(yTrain, leftSubset);
			Lab_t rightAvg = StatisticsHelper::mean(yTrain, rightSubset);
			xt::row(intermediateLabels, leftSon) = xt::zeros<Lab_t>(yTrain.shape());
			xt::row(intermediateLabels, rightSon) = xt::zeros<Lab_t>(yTrain.shape());
			for (size_t sample = 0; sample < yTrain.shape(0); ++sample) {
				intermediateLabels(leftSon, sample) = intermediateLabels(absoluteNode, sample) - leftAvg;
				intermediateLabels(rightSon, sample) = intermediateLabels(absoluteNode, sample) - rightAvg;
			}
		}
		features[h] = bestFeature;
		broCount <<= 1;  // it equals *= 2
	}

	// all internal nodes created
	Lab_t curSum;
	size_t curCnt;
	for (size_t leaf = 0; leaf < leafCnt; ++leaf) {
		curSum = 0;
		curCnt = 0;
		for (auto& sample : subset[innerNodes + leaf]) {
			curSum += yTrain(sample);
		}
		curCnt = subset[innerNodes + leaf].size();
		if (curCnt != 0)
			leaves[leaf] = learningRate * curSum / curCnt;  // mean leaf residual
	}
	// remember or compile tree (in case of JIT compilation enabled)
	treeHolder->newTree(features, thresholds, leaves);

	// free memory
	if (features) {
		delete[] features;
		features = nullptr;
	}
	if (thresholds) {
		delete[] thresholds;
		thresholds = nullptr;
	}
	if (leaves) {
		delete[] leaves;
		leaves = nullptr;
	}
}
