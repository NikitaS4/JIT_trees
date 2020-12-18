#ifndef GBHIST_H
#define GBHIST_H

#include "PybindHeader.h"
#include "Structs.h"
#include <vector>


class GBHist {
public:
	GBHist(const size_t binCount, const pytensor1& xFeature);

	size_t getBinCount() const;
	Lab_t findBestSplit(const pytensor1& xData,
		const std::vector<size_t>& subset, 
		const pytensorY& labels, FVal_t& threshold) const;
	std::vector<size_t> performSplit(const pytensor1& xData,
	const std::vector<size_t>& subset, const FVal_t threshold, 
	std::vector<size_t>& rightSubset) const;
private:
	size_t binCount;
	std::vector<FVal_t> thresholds;

	// functions
	static inline Lab_t square(const Lab_t arg);
	inline size_t whichBin(const FVal_t& sample) const;
};

#endif // GBHIST_H

