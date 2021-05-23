#ifndef GBHIST_H
#define GBHIST_H

#include "PybindHeader.h"
#include "Structs.h"
#include <vector>


class GBHist {
public:
	GBHist(const size_t binCountMin, const size_t binCountMax,
		const size_t treesInEnsemble, const pytensor1& xFeature,
		const Lab_t regularizationParam, const bool randThreshold);

	size_t getBinCount() const;
	Lab_t findBestSplit(const pytensor1& xData,
		const std::vector<size_t>& subset, 
		const pytensorY& labels, FVal_t& threshold);
	std::vector<size_t> performSplit(const pytensor1& xData,
	const std::vector<size_t>& subset, const FVal_t threshold, 
	std::vector<size_t>& rightSubset) const;
	void updateNet(); // add 1 bin each M iterations
	void removeRegularization();
private:
	size_t binCount;
	size_t binCountMin;
	size_t binCountMax;
	size_t binDiff; // how many bins to add
	size_t itersToUpdate; // how many trees will be built with the current hist net
	size_t itersToStopUpdate; // how many trees will be built until updates stop
	size_t itersGone; // the current number of trees
	FVal_t featureMin;
	FVal_t featureMax;
	Lab_t regularizationParam;
	bool randThreshold;
	std::vector<FVal_t> thresholds;
	std::vector<Lab_t> binValue;
	std::vector<size_t> binSize;

	// functions
	static inline Lab_t square(const Lab_t arg);
	static inline FVal_t randomFromInterval(const FVal_t from,
		const FVal_t to);
	inline size_t whichBin(const FVal_t& sample) const;
	inline void updateThresholds();
	inline void fillArraysWithNulls();
};

#endif // GBHIST_H

