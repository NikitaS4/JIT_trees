#ifndef GBDECISION_TREE_H
#define GBDECISION_TREE_H

#include "PybindHeader.h"
#include "Structs.h"
#include "GBHist.h"
#include "../TreeHolders/TreeHolder.h"
#include <vector>


// the class is helper (no instances needed)
class GBDecisionTree {
public:
	GBDecisionTree(const size_t treesInEnsemble,
		const Lab_t regularizationParam);

	~GBDecisionTree();

	static void initStaticMembers(const float learnRate, 
		const size_t trainLen,
		const size_t depth = defaultTreeDepth);
	
	// growTree == FIT
	void growTree(const pytensor2& xTrain,
		const std::vector<size_t>& chosen, 
		const pytensorY& yTrain,
		const std::vector<GBHist>& hists,
		const std::vector<size_t>& featureSubset,
		TreeHolder* treeHolder);

private:
	// tree with depth 1 is node with 2 children
	// leaves = 2 ** height
	// internal nodes = 2 ** (height + 1) - leaves	

	// fields
	float randWeight;
	float weightDelta;
	Lab_t regParam; // regularization parameter
	size_t* features = nullptr;
	FVal_t* thresholds = nullptr;
	Lab_t* leaves = nullptr;

	// methods
	inline FVal_t getSpoiledScore(const FVal_t splitScore) const;

	// static
	static size_t featureCount;
	static size_t treeDepth;
	static bool depthAssigned;
	static size_t innerNodes;
	static size_t leafCnt;
	static float learningRate;
	static std::vector<std::vector<size_t>> subset;

	// constants
	static const size_t defaultTreeDepth = 6;
	static const float scoreInRandNoiseMult;
};

#endif // GBDECISION_TREE_H
