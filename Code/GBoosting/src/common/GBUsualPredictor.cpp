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
    pytensorY answers = treeHolder.predictAllTrees2d(x);
    // don't forget zero predictor (constant)
    for (size_t i = 0; i < answers.shape(0); ++i) {
        answers(i) += zeroPredictor;
    }
    return answers;
}


void GBUsualPredictor::predictTreeTrain(const size_t treeNum) {
    treeHolder.predictTreeFit(xTrain, xValid, treeNum,
        residuals, preds, validRes, validPreds);
}
