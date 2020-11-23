#ifndef GBHIST_H
#define GBHIST_H
#pragma once

#include "Structs.h"
#include <vector>


class GBHist {
public:
	GBHist(const size_t binCount,
		const std::vector<size_t>& sortedIdxs,
		const std::vector<size_t>& backIdxs,
		const std::vector<FVal_t>& xFeature);

	size_t getBinCount() const;
	size_t getDataSplitIdx(const size_t splitPos) const;
	Lab_t findBestSplit(const std::vector<size_t>& subset, 
		const std::vector<Lab_t>& labels, size_t& splitPos) const;
	std::vector<size_t> performSplit(const std::vector<size_t>& subset, 
		const size_t splitPos, 
		std::vector<size_t>& rightSubset) const;
private:
	size_t binCount;
	std::vector<size_t> backIdxs;
	std::vector<size_t> sortedIdxs;
	//std::vector<x_t> thresholds;  // don't needed
	std::vector<size_t> thresholdPos;

	// functions
	size_t binSearch(const std::vector<FVal_t>& xFeature, 
		const FVal_t threshold, const size_t curLeft) const;
};

#endif // GBHIST_H

