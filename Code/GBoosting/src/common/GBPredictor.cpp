#include "GBPredictor.h"
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
    return new GBPredictor(zeroPredictor, treeHolder,
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
    return new GBPredictor(zeroPredictor, treeHolder, xTrain, xValid,
        residuals, preds, validRes, validPreds);
}


Lab_t GBPredictor::predict1d(const pytensor1& x) const {
    validateFeatureCount(x);
	return zeroPredictor + treeHolder.predictAllTrees(x);
}


pytensorY GBPredictor::predict2d(const pytensor2& x) {
    validateFeatureCount(x);
    pytensorY answers = treeHolder.predictAllTrees2d(x);
    // don't forget zero predictor (constant)
    for (size_t i = 0; i < answers.shape(0); ++i) {
        answers(i) += zeroPredictor;
    }
    return answers;
}


void GBPredictor::predictTreeTrain(const size_t treeNum) {
    if (xTrain == nullptr) {
        // it's the case we loaded model and don't need train
        throw std::runtime_error("Can't fit loaded model");
    }
    treeHolder.predictTreeFit(*xTrain, *xValid, treeNum,
        *residuals, *preds, *validRes, *validPreds);
}


void GBPredictor::validateFeatureCount(const pytensor1& x) const {
    if (x.shape(0) != featureCount)
		throw std::runtime_error("Wrong feature count in x_test");
}


void GBPredictor::validateFeatureCount(const pytensor2& x) const {
    if (x.shape(1) != featureCount)
        throw std::runtime_error("Wrong feature count in x_test");
}
