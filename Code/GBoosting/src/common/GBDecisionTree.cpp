#include "GBDecisionTree.h"
#include "StatisticsHelper.h"

#include <stdexcept>


// static initializations
bool GBDecisionTree::depthAssigned = false;
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

GBDecisionTree::GBDecisionTree(const pytensor2& xTrain,
	const std::vector<size_t>& chosen, 
	const pytensorY& yTrain,
	const std::vector<GBHist>& hists) {
	if (!depthAssigned)
		throw std::runtime_error("Tree depth was not assigned");
	features = new size_t[treeDepth];
	thresholds = new FVal_t[innerNodes];
	leaves = new Lab_t[leafCnt];
	pytensor2Y intermediateLabels = xt::zeros<Lab_t>({innerNodes + leafCnt, xTrain.shape(0)});

	// 1st dim - node number, 2nd dim - sample idx
	subset[0] = chosen;	
	xt::row(intermediateLabels, 0) = yTrain;

	size_t featureCount = xTrain.shape(1);

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
		leaves[leaf] = learningRate * curSum / curCnt;  // mean leaf residual
	}
}

GBDecisionTree::GBDecisionTree(GBDecisionTree&& other) noexcept:
	features(std::move(other.features)), 
	thresholds(std::move(other.thresholds)), 
	leaves(std::move(other.leaves)) {
	other.features = nullptr;
	other.thresholds = nullptr;
	other.leaves = nullptr;
}

GBDecisionTree::GBDecisionTree(const GBDecisionTree& other) {
	features = new size_t[treeDepth];
	thresholds = new FVal_t[innerNodes];
	leaves = new Lab_t[leafCnt];

	for (size_t i = 0; i < treeDepth; ++i) {
		features[i] = other.features[i];
	}
	for (size_t i = 0; i < innerNodes; ++i) {
		thresholds[i] = other.thresholds[i];
	}
	for (size_t i = 0; i < leafCnt; ++i) {
		leaves[i] = other.leaves[i];
	}
}

Lab_t GBDecisionTree::predictSingle(const pytensor1& sample) const {
	size_t curNode = 0;
	for (size_t h = 0; h < treeDepth; ++h) {
		if (sample(features[h]) < thresholds[curNode])
			curNode = 2 * curNode + 1;
		else
			curNode = 2 * curNode + 2;
	}
	// now curNode is the node-leaf number
	// have to convert node-leaf number to pure leaf number
	return leaves[curNode - innerNodes];
}

pytensorY GBDecisionTree::predict(const pytensor2& samples) const {
	size_t predCount = samples.shape(0);
	pytensorY preds = xt::zeros<Lab_t>({predCount});
	for (size_t i = 0; i < predCount; ++i) {
		preds(i) = predictSingle(xt::row(samples, i));
	}
	return preds;
}

GBDecisionTree::~GBDecisionTree() {
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
