#include "History.h"

History::History() {}

History::History(const size_t treeNumber,
	const std::vector<Lab_t>& trainLosses,
	const std::vector<Lab_t>& validLosses) : treesLearnt(treeNumber),
	trainLosses(trainLosses), validLosses(validLosses) {}

void History::addLosses(const Lab_t train, const Lab_t valid) {
	trainLosses.push_back(train);
	validLosses.push_back(valid);
}

void History::addAllLosses(const std::vector<Lab_t>& train,
	const std::vector<Lab_t>& valid) {
	trainLosses = train;
	validLosses = valid;
}

void History::setTreesLearnt(const size_t learnt) {
	treesLearnt = learnt;
}

size_t History::getTreesLearnt() const {
	return treesLearnt;
}

std::vector<Lab_t> History::getTrainLosses() const {
	return trainLosses;
}

std::vector<Lab_t> History::getValidLosses() const {
	return validLosses;
}