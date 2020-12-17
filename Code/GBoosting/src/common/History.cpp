#include "History.h"

History::History() {}

History::History(const size_t treeNumber,
	const pyarrayY& trainLosses,
	const pyarrayY& validLosses) : treesLearnt(treeNumber),
	trainLosses(trainLosses), validLosses(validLosses) {}


void History::addAllLosses(const pyarrayY& train,
	const pyarrayY& valid) {
	trainLosses = train;
	validLosses = valid;
}

void History::setTreesLearnt(const size_t learnt) {
	treesLearnt = learnt;
}

size_t History::getTreesLearnt() const {
	return treesLearnt;
}

pyarrayY History::getTrainLosses() const {
	return trainLosses;
}

pyarrayY History::getValidLosses() const {
	return validLosses;
}