#ifndef HISTORY_H
#define HISTORY_H

#include <vector>
#include "PybindHeader.h"
#include "Structs.h"


class History {
public:
	History();
	History(const size_t treeNumber,
		const pytensorY& trainLosses,
		const pytensorY& validLosses);

	// setters
	void addAllLosses(const pytensorY& train,
		const pytensorY& valid);
	void setTreesLearnt(const size_t learnt);

	// getters
	size_t getTreesLearnt() const;
	pytensorY getTrainLosses() const;
	pytensorY getValidLosses() const;

private:
	size_t treesLearnt = 0;
	pytensorY trainLosses;
	pytensorY validLosses;
};

#endif // HISTORY_H

