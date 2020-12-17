#ifndef HISTORY_H
#define HISTORY_H

#include <vector>
#include "PybindHeader.h"
#include "Structs.h"


class History {
public:
	History();
	History(const size_t treeNumber,
		const pyarrayY& trainLosses,
		const pyarrayY& validLosses);

	// setters
	void addAllLosses(const pyarrayY& train,
		const pyarrayY& valid);
	void setTreesLearnt(const size_t learnt);

	// getters
	size_t getTreesLearnt() const;
	pyarrayY getTrainLosses() const;
	pyarrayY getValidLosses() const;

private:
	size_t treesLearnt = 0;
	pyarrayY trainLosses;
	pyarrayY validLosses;
};

#endif // HISTORY_H

