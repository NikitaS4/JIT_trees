#include "GBHist.h"
#include "StatisticsHelper.h"

#include <cmath>


GBHist::GBHist(const size_t binCount, 
	const std::vector<size_t>& sortedIdxs,
	const pytensor1& xFeature): binCount(binCount) {
	size_t n = xFeature.shape(0); // data size
	FVal_t featureMin = xFeature(sortedIdxs[0]);
	FVal_t featureMax = xFeature(sortedIdxs[n - 1]);
	FVal_t binWidth = (featureMax - featureMin) / binCount;
	FVal_t curThreshold;

	size_t curLeft = 0; // left border for the binary search
	size_t posInData = 0; // index in data
	idxToBin = std::vector<size_t>(n, 0);
	for (size_t i = 0; i < binCount; ++i) {
		curThreshold = (i + 1) * binWidth + featureMin; // compute threshold
		thresholds.push_back(curThreshold); // remember threshold
		
		while (posInData < n && xFeature(sortedIdxs[posInData]) < curThreshold) {
			idxToBin[sortedIdxs[posInData++]] = i; // put this one into the current bin			
			// and increase posInData index
		}
	}
}


size_t GBHist::getBinCount() const {
	return binCount;
}


Lab_t GBHist::findBestSplit(const pytensor1& xData,
	const std::vector<size_t>& subset, 
	const pytensorY& labels, FVal_t& threshold) const {
	size_t nSub = subset.size(); // size of the subset

	// compute histograms
	std::vector<Lab_t> binValue(binCount, 0);
	std::vector<size_t> binSize(binCount, 0);
	size_t currentBin = 0;

	Lab_t leftValue = 0;
	Lab_t rightValue = 0;

	// map subset to bins
	for (auto& curX : subset) {
		currentBin = idxToBin[curX]; // get bin number for the current sample
		binValue[currentBin] += labels(curX); // add value (compute sum)
		++binSize[currentBin]; // to compute avg later
		rightValue += labels(curX); // prepare for the best split searching
	}

	// prepare to find the best split
	Lab_t bestScore = 0;
	size_t bestBinNumber = 0;
	bool firstIter = true;
	Lab_t curScore = 0;
	Lab_t leftScore = 0;
	Lab_t rightScore = 0;

	// find best split
	size_t leftSize = 0;
	size_t rightSize = nSub; // at start, all samples are in the right subset

	for (size_t leftLastBin = 0; leftLastBin < binCount - 1; ++leftLastBin) {
		// skip empty bins
		if (binSize[leftLastBin] == 0) {
			continue;
		}
		// try add bin to the left subset
		leftValue += binValue[leftLastBin]; // increase left subset score
		rightValue -= binValue[leftLastBin]; // decrease right subset score

		leftSize += binSize[leftLastBin]; // increase left subset size
		rightSize -= binSize[leftLastBin]; // decrease right subset size

		// compute current score in several steps
		// step 1: compute leaves
		Lab_t leftAvg = leftValue / leftSize;
		Lab_t rightAvg = rightValue / rightSize;
		
		// step 2: calculate RSS
		FVal_t curThreshold = thresholds[leftLastBin];
		leftScore = rightScore = 0; // init scores for sum
		size_t tmpCnter = 0;
		for (auto& curX : subset) {
			if (xData(curX) < curThreshold) {
				// increase left score
				leftScore += square(leftAvg - labels(curX));
			} else {
				// increase right score
				rightScore += square(rightAvg - labels(curX));
			}
		}
		
		// step 3: compute score
		// score(split) = MSE_left + MSE_right (with weights)
		curScore = leftScore / leftSize + rightScore / rightSize;

		// compare with the best value
		if (firstIter || curScore < bestScore) {
			firstIter = false;
			bestScore = curScore;
			bestBinNumber = leftLastBin;
		}
	}
	// return answers
	threshold = thresholds[bestBinNumber];
	return bestScore;
}


std::vector<size_t> GBHist::performSplit(const pytensor1& xData,
	const std::vector<size_t>& subset, const FVal_t threshold, 
	std::vector<size_t>& rightSubset) const {
	std::vector<size_t> leftSubset;
	rightSubset.clear();
	for (auto& curIdx : subset) {
		if (xData(curIdx) < threshold)
			leftSubset.push_back(curIdx);
		else
			rightSubset.push_back(curIdx);
	}
	return leftSubset;
}


Lab_t GBHist::square(const Lab_t arg) {
	return arg * arg;
}
