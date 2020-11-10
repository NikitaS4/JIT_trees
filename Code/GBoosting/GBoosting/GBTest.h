#pragma once

#include "GBoosting.h"

class GBTest : public GradientBoosting {
public:
	static void testSort(std::vector<FVal_t>& xData,
		std::vector<size_t>& sortedIdxs) {
		sortFeature(xData, sortedIdxs);
	}
};