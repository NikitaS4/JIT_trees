#ifndef GBPREDICTOR_H_INCLUDED
#define GBPREDICTOR_H_INCLUDED

#include "PybindHeader.h"
#include "Structs.h"
#include "../TreeHolders/TreeHolder.h"


class GBPredictor {
public:
    static GBPredictor* create(const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2* xTrain,
        const pytensor2* xValid,
        pytensorY* residuals,
        pytensorY* preds,
        pytensorY* validRes,
        pytensorY* validPreds);

    static GBPredictor* createReady(const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const size_t featureCnt);

    virtual ~GBPredictor();

    Lab_t predict1d(const pytensor1& x) const;

    virtual pytensorY predict2d(const pytensor2& x) = 0;
    virtual void predictTreeTrain(const size_t treeNum) = 0;
protected:
    const size_t trainLen;
    const size_t validLen;
    const size_t featureCount;
    const Lab_t zeroPredictor;
    const TreeHolder& treeHolder;
    const pytensor2* xTrain;
    const pytensor2* xValid;
    pytensorY* residuals;
    pytensorY* preds;
    pytensorY* validRes;
    pytensorY* validPreds;

    void validateFeatureCount(const pytensor1& x) const;
    void validateFeatureCount(const pytensor2& x) const;

    GBPredictor(const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2* xTrain,
        const pytensor2* xValid,
        pytensorY* residuals,
        pytensorY* preds,
        pytensorY* validRes,
        pytensorY* validPreds);

    GBPredictor(const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const size_t featureCnt);
};

#endif // GBPREDICTOR_H_INCLUDED
