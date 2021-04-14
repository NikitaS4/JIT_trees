#include "GBUsualPredictor.h"
#include <stdexcept>


GBUsualPredictor::GBUsualPredictor(const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2& xTrain,
        const pytensor2& xValid,
        pytensorY& residuals,
        pytensorY& preds,
        pytensorY& validRes,
        pytensorY& validPreds): GBPredcitor(zeroPredictor,
        treeHolder, xTrain, xValid, residuals, preds,
        validRes, validPreds) {
    // ctor
}


GBUsualPredictor::~GBUsualPredictor() {
    // dtor
}


pytensorY GBUsualPredictor::predict2d(const pytensor2& x) {
    validateFeatureCount(x);
    size_t sampleCnt = x.shape(0);
    pytensorY ans = xt::zeros<Lab_t>({sampleCnt});
    for (size_t sample = 0; sample < sampleCnt; ++sample) {
        ans(sample) = zeroPredictor + treeHolder.predictAllTrees(xt::row(x, sample));
    }
    return ans;
}


void GBUsualPredictor::predictTreeTrain(const size_t treeNum) {
    treeHolder.predictTreeFit(xTrain, xValid, treeNum,
        residuals, preds, validRes, validPreds);
}
