#include "GBDecisionTree.h"


const size_t GBDecisionTree::innerNodes = (1 << treeDepth) - 1;
const size_t GBDecisionTree::leafCnt = 1 << treeDepth;

GBDecisionTree::GBDecisionTree(const std::vector<std::vector<FVal_t>>& xSwapped,
	const std::vector<size_t>& chosen, 
	const std::vector<Lab_t>& yTest,
	const std::vector<GBHist>& hists) {
	features = new size_t[treeDepth];
	thresholds = new FVal_t[innerNodes];
	leaves = new Lab_t[leafCnt];

	// 1st dim - node number, 2nd dim - sample idx
	std::vector<std::vector<size_t>> subset;
	subset.push_back(chosen);

	size_t featureCount = xSwapped.size();

	size_t broCount = 1;
	std::vector<size_t> bestSplitPos;
	Lab_t bestScore;
	bool firstSplitFound = false;
	Lab_t curScore;
	std::vector<size_t> curSplitPos(leafCnt, 0);
	size_t atomicSplitPos;
	size_t bestFeature = 0;
	for (size_t h = 0; h < treeDepth; ++h) {
		// find best split
		size_t firstBroNum = (1 << h) - 1;
		for (size_t feature = 0; feature < featureCount; ++feature) {
			// for all nodes look for the best split of the feature
			curScore = 0;
			for (size_t node = 0; node < broCount; ++node) {
				// find best score
				curScore += hists[feature].findBestSplit(subset[firstBroNum + node], 
					yTest, atomicSplitPos);
				curSplitPos[node] = atomicSplitPos;
			}
			if (!firstSplitFound || curScore < bestScore) {
				bestScore = curScore;
				bestSplitPos = curSplitPos;
				bestFeature = feature;
			}
		}
		// the best score is found now
		// need to perform the split
		for (size_t node = 0; node < broCount; ++node) {
			std::vector<size_t> leftSubset;
			std::vector<size_t> rightSubset;
			hists[bestFeature].performSplit(subset[firstBroNum + node],
				bestSplitPos[node], leftSubset, rightSubset);
			// subset will be placed to their topological places
			subset.push_back(leftSubset);
			subset.push_back(rightSubset);
			thresholds[firstBroNum + node] = xSwapped[bestFeature][hists[bestFeature].getDataSplitIdx(bestSplitPos[node])];
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
			curSum += yTest[sample];
		}
		curCnt = subset[innerNodes + leaf].size();
		leaves[leaf] = curSum / curCnt;  // mean leaf residual
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

Lab_t GBDecisionTree::predict(const std::vector<FVal_t>& sample) {
	size_t curNode = 0;
	for (size_t h = 0; h < treeDepth; ++h) {
		if (sample[features[h]] < thresholds[curNode])
			curNode = curNode * 2 + 1;
		else
			curNode = curNode * 2 + 2;
	}
	// now curNode is the node-leaf number
	// have to convert node-leaf number to pure leaf number
	return leaves[curNode - innerNodes];
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