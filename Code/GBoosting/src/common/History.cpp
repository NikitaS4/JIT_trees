#include "History.h"

History::History() {}

History::History(const size_t treeNumber,
	const pytensorY& trainLosses,
	const pytensorY& validLosses) : treesLearnt(treeNumber),
	trainLosses(trainLosses), validLosses(validLosses) {}


void History::addAllLosses(const pytensorY& train,
	const pytensorY& valid) {
	trainLosses = train;
	validLosses = valid;
}

void History::setTreesLearnt(const size_t learnt) {
	treesLearnt = learnt;
}

size_t History::getTreesLearnt() const {
	return treesLearnt;
}

pytensorY History::getTrainLosses() const {
	return trainLosses;
}

pytensorY History::getValidLosses() const {
	return validLosses;
}