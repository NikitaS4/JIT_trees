#include "GBDecisionTree.h"
#include "StatisticsHelper.h"

#include <stdexcept>
#include <cstdlib>


// static initializations
bool GBDecisionTree::depthAssigned = false;
size_t GBDecisionTree::featureCount = 0;
size_t GBDecisionTree::treeDepth = 0;
size_t GBDecisionTree::innerNodes = 0;
size_t GBDecisionTree::leafCnt = 0;
float GBDecisionTree::learningRate = 1;
std::vector<std::vector<size_t>> GBDecisionTree::subset;
const float GBDecisionTree::scoreInRandNoiseMult = 0.3f;


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


GBDecisionTree::GBDecisionTree(const size_t treesInEnsemble,
	const Lab_t regularizationParam): 
	randWeight(1.0f),
	weightDelta(2.0f / float(treesInEnsemble)),
	regParam(regularizationParam) {
		// init memory for the buffers
		features = features = new size_t[treeDepth];
		thresholds = new FVal_t[innerNodes];
		leaves = new Lab_t[leafCnt];
		// allocate memory for the thresholds array
		curThreshold = std::vector<FVal_t>(leafCnt, 0);
		bestThreshold = std::vector<FVal_t>(leafCnt, 0);
}


GBDecisionTree::~GBDecisionTree() {
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


void GBDecisionTree::growTree(const pytensor2& xTrain,
	const std::vector<size_t>& chosen, 
	const pytensorY& yTrain,
	const std::vector<size_t>& featureSubset,
	std::vector<GBHist>& hists,
	TreeHolder* treeHolder) {
	if (!depthAssigned)
		throw std::runtime_error("Tree depth was not assigned");
	for (size_t i = 0; i < innerNodes; ++i)
		thresholds[i] = 0;
	for (size_t i = 0; i < leafCnt; ++i)
		leaves[i] = 0;
	pytensor2Y intermediateLabels = xt::zeros<Lab_t>({innerNodes + leafCnt, xTrain.shape(0)});

	// 1st dim - node number, 2nd dim - sample idx
	subset[0] = chosen;	
	xt::row(intermediateLabels, 0) = yTrain;

	featureCount = xTrain.shape(1);
	size_t featureSubCount = featureSubset.size();

	size_t broCount = 1;
	Lab_t bestScore;
	bool firstSplitFound = false;
	Lab_t curScore;
	FVal_t atomicThreshold;
	size_t bestFeature = 0;
	for (size_t h = 0; h < treeDepth; ++h) {
		// find best split
		size_t firstBroNum = (1 << h) - 1;
		for (size_t curFeature = 0; curFeature < featureSubCount; ++curFeature) {
			// for all nodes look for the best split of the feature
			size_t feature = featureSubset[curFeature]; // get current feature from subset

			curScore = 0;
			for (size_t node = 0; node < broCount; ++node) {
				// find best score
				curScore += hists[feature].findBestSplit(xt::col(xTrain, feature),
					subset[firstBroNum + node], 
					xt::row(intermediateLabels, firstBroNum + node), atomicThreshold);
				curThreshold[node] = atomicThreshold;
			}
			// add random noise to the score
			// this will make the chosen tree split to be
			// not as optimal as it could be
			// this diminishes overfitiing of the ensemble
			curScore = getSpoiledScore(curScore);
			if (!firstSplitFound || curScore < bestScore) {
				bestScore = curScore;
				cpyThresholds(); // bestThreshold = curThreshold
				bestFeature = feature;
				firstSplitFound = true;
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
	// each leaf will be multiplied onto learning rate and regLeafUnit
	for (size_t leaf = 0; leaf < leafCnt; ++leaf) {
		curSum = 0;
		curCnt = 0;
		for (auto& sample : subset[innerNodes + leaf]) {
			curSum += yTrain(sample);
		}
		curCnt = subset[innerNodes + leaf].size();
		if (curCnt != 0)
			leaves[leaf] = learningRate * curSum / (regParam + curCnt);  // mean leaf residual
	}
	// remember or compile tree (in case of JIT compilation enabled)
	treeHolder->newTree(features, thresholds, leaves);

	// update randWeight (for the next tree)
	randWeight -= weightDelta;
}


void GBDecisionTree::removeRegularization() {
	regParam = 0;
}


FVal_t GBDecisionTree::getSpoiledScore(const FVal_t splitScore) const {
	// generate random value in [0; 1)
	float noise = float(std::rand()) / (float(1) + RAND_MAX);
	// rescale
	noise *= scoreInRandNoiseMult * splitScore * randWeight;
	return splitScore + noise;
}


void GBDecisionTree::cpyThresholds() {
	// copy curThreshold to the bestThreshold
	for (size_t i = 0; i < leafCnt; ++i) {
		bestThreshold[i] = curThreshold[i];
	}
}
