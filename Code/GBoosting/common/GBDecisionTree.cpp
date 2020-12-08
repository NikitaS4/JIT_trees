#include "GBDecisionTree.h"
#include "StatisticsHelper.h"

#include <stdexcept>
#include <iostream>


// static initializations
bool GBDecisionTree::depthAssigned = false;
size_t GBDecisionTree::treeDepth = 0;
size_t GBDecisionTree::innerNodes = 0;
size_t GBDecisionTree::leafCnt = 0;
float GBDecisionTree::learningRate = 1;
std::vector<std::vector<size_t>> GBDecisionTree::subset;
std::vector<std::vector<Lab_t>> GBDecisionTree::intermediateLabels;


void GBDecisionTree::initStaticMembers(const float learnRate,
	const size_t depth) {
	if (depth == 0)
		throw std::runtime_error("Wrong tree depth");
	treeDepth = depth;
	innerNodes = (1 << treeDepth) - 1;
	leafCnt = 1 << treeDepth;		
	subset = std::vector<std::vector<size_t>>(innerNodes + leafCnt,
		std::vector<size_t>());
	intermediateLabels = std::vector<std::vector<Lab_t>>(innerNodes + leafCnt, 
	std::vector<Lab_t>());
	depthAssigned = true;
	learningRate = learnRate;
}

GBDecisionTree::GBDecisionTree(const std::vector<std::vector<FVal_t>>& xSwapped,
	const std::vector<size_t>& chosen, 
	const std::vector<Lab_t>& yTrain,
	const std::vector<GBHist>& hists) {
	if (!depthAssigned)
		throw std::runtime_error("Tree depth was not assigned");
	features = new size_t[treeDepth];
	thresholds = new FVal_t[innerNodes];
	leaves = new Lab_t[leafCnt];

	// 1st dim - node number, 2nd dim - sample idx
	subset[0] = chosen;	
	intermediateLabels[0] = yTrain;

	size_t featureCount = xSwapped.size();

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
				curScore += hists[feature].findBestSplit(xSwapped[feature],
					subset[firstBroNum + node], 
					intermediateLabels[firstBroNum + node], atomicThreshold);
				curThreshold[node] = atomicThreshold;
			}
			if (!firstSplitFound || curScore < bestScore) {
				bestScore = curScore;
				bestThreshold = curThreshold;
				bestFeature = feature;
			}
		}
		// the best score is found now
		//usedFeatures.push_back(bestFeature);  // lock feature
		// need to perform the split
		for (size_t node = 0; node < broCount; ++node) {
			std::vector<size_t> leftSubset;
			std::vector<size_t> rightSubset;
			size_t absoluteNode = firstBroNum + node;
			thresholds[absoluteNode] = bestThreshold[node];
			leftSubset = hists[bestFeature].performSplit(xSwapped[bestFeature],
				subset[absoluteNode], bestThreshold[node], rightSubset);
			// subset will be placed to their topological places
			size_t leftSon = 2 * absoluteNode + 1;
			size_t rightSon = leftSon + 1;
			subset[leftSon] = leftSubset;
			subset[rightSon] = rightSubset;
			// update labels
			Lab_t leftAvg = StatisticsHelper::mean(yTrain, leftSubset);
			Lab_t rightAvg = StatisticsHelper::mean(yTrain, rightSubset);
			intermediateLabels[leftSon].clear();
			intermediateLabels[rightSon].clear();
			for (size_t sample = 0; sample < yTrain.size(); ++sample) {
				intermediateLabels[leftSon].push_back(intermediateLabels[absoluteNode][sample] - leftAvg);
				intermediateLabels[rightSon].push_back(intermediateLabels[absoluteNode][sample] - rightAvg);
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
			curSum += yTrain[sample];
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

Lab_t GBDecisionTree::predict(const std::vector<FVal_t>& sample) const {
	size_t curNode = 0;
	for (size_t h = 0; h < treeDepth; ++h) {
		if (sample[features[h]] < thresholds[curNode])
			curNode = 2 * curNode + 1;
		else
			curNode = 2 * curNode + 2;
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

void GBDecisionTree::printTree() const {
	std::cout << "Printing GBDT\n\n";
	
	std::cout << "h = " << treeDepth << "\n";
	size_t broCount = 1;
	size_t curNode = 0;
	for (size_t depth = 0; depth < treeDepth; ++depth) {
		std::cout << "Feature: " << features[depth] << "\n";
		std::cout << "Thresholds: ";
		for (size_t node = 0; node < broCount; ++node) {
			std::cout << thresholds[curNode] << " ";
		}
		std::cout << "\n";
		broCount *= 2;
		++curNode;
	}

	std::cout << "Leaves: ";
	for (size_t leaf = 0; leaf < leafCnt; ++leaf) {
		std::cout << leaves[leaf] << " ";
	}
	std::cout << "\n\n";
}