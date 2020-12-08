#ifndef GBHIST_H
#define GBHIST_H

#include "Structs.h"
#include <vector>


class GBHist {
public:
	GBHist(const size_t binCount,
		const std::vector<size_t>& sortedIdxs,
		const std::vector<FVal_t>& xFeature);

	size_t getBinCount() const;
	Lab_t findBestSplit(const std::vector<FVal_t>& xData,
		const std::vector<size_t>& subset, 
		const std::vector<Lab_t>& labels, FVal_t& threshold) const;
	std::vector<size_t> performSplit(const std::vector<FVal_t>& xData,
	const std::vector<size_t>& subset, const FVal_t threshold, 
	std::vector<size_t>& rightSubset) const;
private:
	size_t binCount;
	std::vector<FVal_t> thresholds;  // don't needed ?
	std::vector<size_t> idxToBin; // index in X to bin number
	std::vector<size_t> thresholdPos;

	// functions
	static inline Lab_t square(const Lab_t arg);
};

#endif // GBHIST_H

