#ifndef GBDECISION_TREE_H
#define GBDECISION_TREE_H

#include "PybindHeader.h"
#include "Structs.h"
#include "GBHist.h"
#include "TreeHolder.h"
#include <vector>


// the class is helper (no instances needed)
class GBDecisionTree {
public:
	GBDecisionTree(const size_t treesInEnsemble,
		const Lab_t regularizationParam,
		const bool spoilScores,
		const float learningRate,
		const size_t trainLen, const size_t depth);

	~GBDecisionTree();
	
	// growTree == FIT
	void growTree(const pytensor2& xTrain,
		const std::vector<size_t>& chosen, 
		const pytensorY& yTrain,
		const std::vector<size_t>& featureSubset,
		std::vector<GBHist>& hists,
		TreeHolder* treeHolder);

	void removeRegularization();

private:
	// tree with depth 1 is node with 2 children
	// leaves = 2 ** height
	// internal nodes = 2 ** (height + 1) - leaves	

	// fields
	float randWeight;
	float weightDelta;
	Lab_t regParam; // regularization parameter
	std::vector<FVal_t> curThreshold;
	std::vector<FVal_t> bestThreshold;
	std::vector<size_t> features;
	std::vector<FVal_t> thresholds;
	std::vector<Lab_t> leaves;
	bool spoilScores;
	size_t featureCount;
	size_t treeDepth;
	size_t innerNodes;
	size_t leafCnt;
	float learningRate;
	std::vector<std::vector<size_t>> subset;

	// methods
	inline FVal_t getSpoiledScore(const FVal_t splitScore) const;
	inline void cpyThresholds(); // copy curThreshold to the bestThreshold
	inline void validateTree();

	// constants
	static const float scoreInRandNoiseMult;
};

#endif // GBDECISION_TREE_H
