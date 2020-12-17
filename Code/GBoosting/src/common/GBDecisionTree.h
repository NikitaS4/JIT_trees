#ifndef GBDECISION_TREE_H
#define GBDECISION_TREE_H

#include "PybindHeader.h"
#include "Structs.h"
#include "GBHist.h"
#include <vector>


class GBDecisionTree {
public:
	// CTOR == FIT
	GBDecisionTree(const pytensor2& xTrain,
		const std::vector<size_t>& chosen, 
		const pytensorY& yTrain,
		const std::vector<GBHist>& hists);
	GBDecisionTree(GBDecisionTree&& other) noexcept;  // move ctor
	GBDecisionTree(const GBDecisionTree& other);  // copy ctor
	virtual ~GBDecisionTree();

	Lab_t predict(const pytensor1& sample) const;

	static void initStaticMembers(const float learnRate, 
		const size_t trainLen,
		const size_t depth = defaultTreeDepth);

private:
	// tree with depth 1 is node with 2 children
	// leaves = 2 ** height
	// internal nodes = 2 ** (height + 1) - leaves
	size_t* features = nullptr;
	FVal_t* thresholds = nullptr;
	Lab_t* leaves = nullptr;

	// static
	static size_t treeDepth;
	static bool depthAssigned;
	static size_t innerNodes;
	static size_t leafCnt;
	static float learningRate;
	static std::vector<std::vector<size_t>> subset;

	// constants
	static const size_t defaultTreeDepth = 6;
};

#endif // GBDECISION_TREE_H
