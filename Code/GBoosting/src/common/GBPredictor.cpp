#include "GBPredictor.h"
#include "GBUsualPredictor.h"
#include <stdexcept>


GBPredictor::~GBPredictor() {
    // dtor
}


GBPredictor::GBPredictor(const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2* xTrain,
        const pytensor2* xValid,
        pytensorY* residuals,
        pytensorY* preds,
        pytensorY* validRes,
        pytensorY* validPreds):
        trainLen((xTrain)? (xTrain->shape(0)) : (0)),
        featureCount((xTrain)? (xTrain->shape(1)) : (0)),
        validLen((xValid)? (xValid->shape(0)) : (0)),
        zeroPredictor(zeroPredictor), treeHolder(treeHolder),
        xTrain(xTrain), xValid(xValid),
        residuals(residuals), preds(preds),
        validRes(validRes), validPreds(validPreds) {
    // ctor
}


GBPredictor::GBPredictor(const Lab_t zeroPredictor, const TreeHolder& treeHolder,
    const size_t featureCnt): trainLen(0), featureCount(featureCnt), validLen(0),
    zeroPredictor(zeroPredictor), treeHolder(treeHolder),
    xTrain(nullptr), xValid(nullptr), residuals(nullptr),
    preds(nullptr), validRes(nullptr), validPreds(nullptr) {
    // ctor
}


GBPredictor* GBPredictor::createReady(const Lab_t zeroPredictor,
    const TreeHolder& treeHolder, const size_t featureCnt) {
    return new GBUsualPredictor(zeroPredictor, treeHolder,
        featureCnt);
}


GBPredictor* GBPredictor::create(const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2* xTrain,
        const pytensor2* xValid,
        pytensorY* residuals,
        pytensorY* preds,
        pytensorY* validRes,
        pytensorY* validPreds) {
    return new GBUsualPredictor(zeroPredictor, treeHolder, xTrain, xValid,
        residuals, preds, validRes, validPreds);
}


Lab_t GBPredictor::predict1d(const pytensor1& x) const {
    validateFeatureCount(x);
	return zeroPredictor + treeHolder.predictAllTrees(x);
}


void GBPredictor::validateFeatureCount(const pytensor1& x) const {
    if (x.shape(0) != featureCount)
		throw std::runtime_error("Wrong feature count in x_test");
}


void GBPredictor::validateFeatureCount(const pytensor2& x) const {
    if (x.shape(1) != featureCount)
        throw std::runtime_error("Wrong feature count in x_test");
}
