#ifndef GBHIST_H
#define GBHIST_H

#include "PybindHeader.h"
#include "Structs.h"
#include <vector>


class GBHist {
public:
	GBHist(const size_t binCount,
		const std::vector<size_t>& sortedIdxs,
		const pyarray& xFeature);

	size_t getBinCount() const;
	Lab_t findBestSplit(const pyarray& xData,
		const std::vector<size_t>& subset, 
		const pyarrayY& labels, FVal_t& threshold) const;
	std::vector<size_t> performSplit(const pyarray& xData,
	const std::vector<size_t>& subset, const FVal_t threshold, 
	std::vector<size_t>& rightSubset) const;
private:
	size_t binCount;
	std::vector<FVal_t> thresholds;
	std::vector<size_t> idxToBin; // index in X to bin number

	// functions
	static inline Lab_t square(const Lab_t arg);
};

#endif // GBHIST_H

