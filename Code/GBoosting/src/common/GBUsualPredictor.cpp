#include "GBUsualPredictor.h"
#include <stdexcept>


GBUsualPredictor::GBUsualPredictor(const size_t threadCnt,
        const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2& xTrain,
        const pytensor2& xValid,
        pytensorY& residuals,
        pytensorY& preds,
        pytensorY& validRes,
        pytensorY& validPreds): GBPredcitor(threadCnt, zeroPredictor,
        treeHolder, xTrain, xValid, residuals, preds,
        validRes, validPreds) {
    // ctor
    if (threadCnt != 1)
        throw std::runtime_error("Wrong thread count passed to GBUsualPredictor");
}


GBUsualPredictor::~GBUsualPredictor() {
    // dtor
}


pytensorY GBUsualPredictor::predict2d(const pytensor2& x) const {
    validateFeatureCount(x);
    size_t sampleCnt = x.shape(0);
    pytensorY ans = xt::zeros<Lab_t>({sampleCnt});
    for (size_t sample = 0; sample < sampleCnt; ++sample) {
        ans(sample) = zeroPredictor + treeHolder.predictAllTrees(xt::row(x, sample));
    }
    return ans;
}


void GBUsualPredictor::predictTreeTrain(const size_t treeNum) {
    // update train residuals
    for (size_t sample = 0; sample < trainLen; ++sample) {
		Lab_t prediction = treeHolder.predictTree(xt::row(xTrain, sample), 
			treeNum);
		residuals(sample) -= prediction;
		preds(sample) += prediction;
	}

    // update validation residuals
	for (size_t sample = 0; sample < validLen; ++sample) {
		Lab_t prediction = treeHolder.predictTree(xt::row(xValid, sample),
			treeNum);
		validRes(sample) -= prediction;
		validPreds(sample) += prediction;
	}
}
