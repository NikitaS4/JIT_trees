#pragma once

#include "Structs.h"
#include "GBHist.h"
#include <vector>

class GBDecisionTree {
public:
	// CTOR == FIT
	GBDecisionTree(const std::vector<std::vector<FVal_t>>& xSwapped,
		const std::vector<size_t>& chosen, 
		const std::vector<Lab_t>& yTrain,
		const std::vector<GBHist>& hists);
	GBDecisionTree(GBDecisionTree&& other) noexcept;  // move ctor
	GBDecisionTree(const GBDecisionTree& other);  // copy ctor
	virtual ~GBDecisionTree();

	Lab_t predict(const std::vector<FVal_t>& sample) const;

	static void initTreeDepth(const size_t depth = defaultTreeDepth);
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
	static std::vector<std::vector<size_t>> subset;

	// constants
	static const size_t defaultTreeDepth = 6;
};