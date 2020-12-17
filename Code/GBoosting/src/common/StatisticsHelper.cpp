#include "StatisticsHelper.h"

#include <cmath>


Lab_t StatisticsHelper::mean(const pyarrayY& vals) {
	size_t count = vals.shape(0);
	Lab_t curSum = 0;
	for (auto& curVal : vals) {
		curSum += curVal;
	}
	return curSum / count;
}

Lab_t StatisticsHelper::mean(const pyarrayY& vals,
	const std::vector<size_t>& idxs) {
	size_t count = idxs.size();
	Lab_t curSum = 0;
	for (auto& curIdx : idxs) {
		curSum += vals[curIdx];
	}
	return curSum / count;
}

Lab_t StatisticsHelper::maxAbs(const pyarrayY& vals) {
	Lab_t curMax = 0;
	for (auto& curVal : vals) {
		if (abs(curVal) > curMax)
			curMax = abs(curVal);
	}
	return curMax;
}
