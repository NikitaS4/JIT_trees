#include "GBHist.h"
#include "StatisticsHelper.h"

#include <cmath>


GBHist::GBHist(const size_t binCount, 
	const std::vector<size_t>& sortedIdxs,
	const std::vector<size_t>& backIdxs,
	const std::vector<FVal_t>& xFeature): binCount(binCount), 
	backIdxs(backIdxs), sortedIdxs(sortedIdxs) {
	FVal_t featureMin = xFeature[sortedIdxs[0]];
	FVal_t featureMax = xFeature[sortedIdxs[sortedIdxs.size() - 1]];
	FVal_t binWidth = (featureMax - featureMin) / binCount;
	FVal_t curThreshold;
	
	size_t curLeft = 0; // left border for the binary search
	for (size_t i = 0; i < binCount; ++i) {
		curThreshold = (i + 1) * binWidth + featureMin;
		//thresholds.push_back(curThreshold);

		// Remember result to use it in search 
		// in the next iteration
		curLeft = binSearch(xFeature, curThreshold, curLeft);
		thresholdPos.push_back(curLeft);
	}
}


size_t GBHist::getBinCount() const {
	return binCount;
}

size_t GBHist::getDataSplitIdx(const size_t splitPos) const {
	return sortedIdxs[splitPos];
}


Lab_t GBHist::findBestSplit(const std::vector<size_t>& subset, 
	const std::vector<Lab_t>& labels, size_t& splitPos) const {
	std::vector<size_t> sortedSubset;
	//std::vector<size_t> backSortedSubset;
	for (auto& curX : subset)
		sortedSubset.push_back(sortedIdxs[curX]);
	size_t subsetSize = sortedSubset.size();
	Lab_t leftSum = 0;
	Lab_t rightSum = 0;
	size_t leftCnt = 0;
	size_t rightCnt = subsetSize;  // at start
	size_t prevPos = 0;

	// at start the right sum is all subset sum
	for (size_t sample = 0; sample < subsetSize; ++sample) {
		rightSum += labels[subset[sample]];
	}

	Lab_t leftAvg;
	Lab_t rightAvg;
	Lab_t leftScore;
	Lab_t rightScore;
	Lab_t curScore;
	bool firstScoreFound = false;
	Lab_t bestScore = 0;
	size_t bestSplitPos = 0;
	// for each bin calculate score
	// new threshold -> left += diff, right -= diff
	size_t sample;
	size_t curBin = 0;
	while (curBin < binCount - 1 && prevPos < subsetSize) {
		// skip empty bins
		while (curBin < binCount - 1 &&
			thresholdPos[curBin] <= subset[prevPos])
			++curBin;
		size_t curThPos = thresholdPos[curBin];
		for (sample = prevPos; sample < subsetSize && subset[sample] < curThPos; ++sample) {
			leftSum += labels[sortedSubset[sample]];
			rightSum -= labels[sortedSubset[sample]];
			++leftCnt;
		}
		rightCnt = subsetSize - leftCnt;  // subsetSize = left + right
		prevPos = sample;  // for the next step
		leftAvg = leftSum / leftCnt;
		rightAvg = rightSum / rightCnt;
		leftScore = 0;
		rightScore = 0;

		// score(split) = MSE_left + MSE_right (with weights)
		for (sample = 0; sample < leftCnt; ++sample) {
			leftScore += (leftAvg - labels[sortedSubset[sample]]) * (leftAvg - labels[sortedSubset[sample]]);
		}
		for (/*sample = leftCnt*/; sample < subsetSize; ++sample) {
			rightScore += (rightAvg - labels[sortedSubset[sample]]) * (rightAvg - labels[sortedSubset[sample]]);
		}
		curScore = leftScore / leftCnt + rightScore / rightCnt;
		if (!firstScoreFound || curScore < bestScore) {
			firstScoreFound = true;
			bestScore = curScore;
			bestSplitPos = curThPos;
		}
		++curBin;  // get the next bin
	}
	// now the best score found	
	splitPos = bestSplitPos;
	return bestScore;
}


std::vector<size_t> GBHist::performSplit(const std::vector<size_t>& subset,
	const size_t splitPos, std::vector<size_t>& rightSubset) const {
	std::vector<size_t> leftSubset;
	rightSubset.clear();
	for (auto& curIdx : subset) {
		if (backIdxs[curIdx] < splitPos)
			leftSubset.push_back(curIdx);
		else
			rightSubset.push_back(curIdx);
	}
	return leftSubset;
}


size_t GBHist::binSearch(const std::vector<FVal_t>& xFeature, 
	const FVal_t threshold, const size_t curLeft) const {
	// binary search
	size_t left = curLeft;  // left most at start
	size_t right = sortedIdxs.size() - 1;  // right most at start
	size_t mid;

	FVal_t leftVal = xFeature[sortedIdxs[left]];
	FVal_t midVal;

	while (leftVal < threshold && (right - left) > 1) {
		mid = (right + left) / 2;
		midVal = xFeature[sortedIdxs[mid]];
		if (threshold >= midVal) {
			leftVal = midVal;
			left = mid;
		}
		else {
			right = mid;
		}
	}

	// leftVal >= threshold, so 'left' is the last in the cur bin
	return left;
}
