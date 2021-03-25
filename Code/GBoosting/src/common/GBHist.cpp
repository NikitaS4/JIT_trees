#include "GBHist.h"
#include "StatisticsHelper.h"

#include <cmath>
#include <cstdlib>


GBHist::GBHist(const size_t binCountMin, const size_t binCountMax, 
	const size_t treesInEnsemble, const pytensor1& xFeature,
	const Lab_t regularizationParam): 
	binCount(binCountMin), binCountMin(binCountMin), 
	binCountMax(binCountMax), itersGone(0),
	regularizationParam(regularizationParam) {
	size_t n = xFeature.shape(0); // data size
	featureMin = xFeature(0); // at start
	featureMax = featureMin;
	for (size_t i = 1; i < n; ++i) { // find min and max
		if (xFeature(i) < featureMin)
			featureMin = xFeature(i);
		if (xFeature(i) > featureMax)
			featureMax = xFeature(i);
	}
	FVal_t binWidth = (featureMax - featureMin) / binCount;
	FVal_t curThreshold;

	for (size_t i = 0; i < binCount; ++i) {
		curThreshold = (i + 1) * binWidth + featureMin; // compute threshold
		thresholds.push_back(curThreshold); // remember threshold
	}
	// Static bin count in histograms
	if (binCountMin == binCountMax) {
		itersToStopUpdate = 0; // don't update at all
		return;
	}

	// Dynamic bin count in histograms
	// Compute itersToUpdate
	const float fitPartWhenBinsMax = 0.7f;
	float itersPerBinIncrement = fitPartWhenBinsMax * treesInEnsemble / float(binCountMax - binCountMin);
	if (itersPerBinIncrement < 1.0f) {
		// binDiff > 1
		binDiff = size_t(1 / itersPerBinIncrement);
		itersToUpdate = 1;
	} else {
		// binDiff == 1 for each itersToUpdate iterations
		binDiff = 1;
		itersToUpdate = size_t(itersPerBinIncrement);
	}
	itersToStopUpdate = binDiff * (binCountMax - binCountMin) / itersToUpdate;
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
		currentBin = whichBin(xData(curX)); // get bin number for the current sample		
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
		// We use regularized MSE/2 as the loss function
		// gradients are f(x_i) - y_i
		// hessians are 1 (const)
		// leftSize (rightSize) is the sum of the hessians
		// leftValue (rightValue) is the sum of the gradients
		Lab_t leftAvg = leftValue / (leftSize + regularizationParam);
		Lab_t rightAvg = rightValue / (rightSize + regularizationParam);
		// now leftAvg (rightAvg) is the leaf weight corresponding to the current node

		// step 2: calculate RSS
		FVal_t curThreshold = thresholds[leftLastBin];
		leftScore = rightScore = 0; // init scores for sum
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
		// score(split) = MSE_left + MSE_right (no weights needed)
		curScore = leftScore + rightScore;

		// compare with the best value
		if (firstIter || curScore < bestScore) {
			firstIter = false;
			bestScore = curScore;
			bestBinNumber = leftLastBin;
		}
	}

	if (bestBinNumber != 0 && bestBinNumber != binCount - 1) {
		// the bucket is somwhere in the middle on the histogram
		// get random threshold from the interval <a, b>, where
		// a - it the left border of the left bucket and
		// b - the right border of the right bucket
		threshold = randomFromInterval(thresholds[bestBinNumber - 1],
			thresholds[bestBinNumber + 1]);
	} else
		// it's the leftmost or the rightmost bucket
		threshold = thresholds[bestBinNumber];
	
	// return answers
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


size_t GBHist::whichBin(const FVal_t& sample) const {
	size_t bin = 0;
	do {
		++bin;
	} while (bin < binCount && sample > thresholds[bin]);
	return bin - 1;
}


FVal_t GBHist::randomFromInterval(const FVal_t from,
		const FVal_t to) {
	return FVal_t(to - from) * FVal_t(rand()) / (FVal_t(1) + RAND_MAX) + from;
}


void GBHist::updateThresholds() {
	if (binCount >= binCountMax)
		return; // don't need recomputing
	// recompute bin count
	binCount += binDiff;
	const size_t currentLength = thresholds.size();
	const size_t tail = binCount - currentLength;
	const FVal_t binWidth = (featureMax - featureMin) / binCount;
	// update thresholds for the allocated part
	for (size_t i = 0; i < currentLength; ++i) {
		thresholds[i] = (i + 1) * binWidth + featureMin;
	}
	// add new thresholds
	for (size_t i = 0; i < tail; ++i) {
		thresholds.push_back((i + currentLength + 1) * binWidth + featureMin);
	}
}


void GBHist::updateNet() {
	++itersGone;
	if (itersGone > itersToStopUpdate) {
		return;
	}
	if (itersGone % itersToUpdate == 0) {
		updateThresholds();
	}
}