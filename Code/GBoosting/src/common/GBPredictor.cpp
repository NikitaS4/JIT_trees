#include "GBPredictor.h"
#include "GBUsualPredictor.h"
#include <stdexcept>


GBPredcitor::~GBPredcitor() {
    // dtor
}


GBPredcitor::GBPredcitor(const size_t threadCnt,
        const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2& xTrain,
        const pytensor2& xValid,
        pytensorY& residuals,
        pytensorY& preds,
        pytensorY& validRes,
        pytensorY& validPreds):
        threadCnt(threadCnt), trainLen(xTrain.shape(0)),
        featureCount(xTrain.shape(1)), validLen(xValid.shape(0)),
        zeroPredictor(zeroPredictor), treeHolder(treeHolder),
        xTrain(xTrain), xValid(xValid),
        residuals(residuals), preds(preds),
        validRes(validRes), validPreds(validPreds) {
    // ctor
}


GBPredcitor* GBPredcitor::create(const size_t threadCnt,
        const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2& xTrain,
        const pytensor2& xValid,
        pytensorY& residuals,
        pytensorY& preds,
        pytensorY& validRes,
        pytensorY& validPreds) {
    if (threadCnt == 0)
        throw std::runtime_error("Wrong thread count (0)");
    return new GBUsualPredictor(threadCnt,
        zeroPredictor, treeHolder, xTrain, xValid,
        residuals, preds, validRes, validPreds);
}


Lab_t GBPredcitor::predict1d(const pytensor1& x) const {
    validateFeatureCount(x);
	return zeroPredictor + treeHolder.predictAllTrees(x);
}


void GBPredcitor::validateFeatureCount(const pytensor1& x) const {
    if (x.shape(0) != featureCount)
		throw std::runtime_error("Wrong feature count in x_test");
}


void GBPredcitor::validateFeatureCount(const pytensor2& x) const {
    if (x.shape(1) != featureCount)
        throw std::runtime_error("Wrong feature count in x_test");
}
