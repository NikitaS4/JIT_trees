#ifndef GBTEST_H
#define GBTEST_H

#include "GBoosting.h"


class GBTest : public GradientBoosting {
public:
	static std::vector<size_t> testSort(std::vector<FVal_t>& xData) {
		return sortFeature(xData);
	}
};

#endif // GBTEST_H

