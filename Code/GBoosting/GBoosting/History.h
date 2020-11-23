#ifndef HISTORY_H
#define HISTORY_H
#pragma once

#include <vector>
#include "Structs.h"


class History {
public:
	History();
	History(const size_t treeNumber,
		const std::vector<Lab_t>& trainLosses,
		const std::vector<Lab_t>& validLosses);

	// setters
	void addLosses(const Lab_t train, const Lab_t valid);
	void addAllLosses(const std::vector<Lab_t>& train,
		const std::vector<Lab_t>& valid);
	void setTreesLearnt(const size_t learnt);

	// getters
	size_t getTreesLearnt() const;
	std::vector<Lab_t> getTrainLosses() const;
	std::vector<Lab_t> getValidLosses() const;

private:
	size_t treesLearnt = 0;
	std::vector<Lab_t> trainLosses;
	std::vector<Lab_t> validLosses;
};

#endif // HISTORY_H

