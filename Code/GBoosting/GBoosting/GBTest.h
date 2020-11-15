#pragma once

#include "GBoosting.h"

class GBTest : public GradientBoosting {
public:
	static std::vector<size_t> testSort(std::vector<FVal_t>& xData) {
		return sortFeature(xData);
	}
};