#ifndef GBUSUAL_PREDICTOR_H_INCLUDED
#define GBUSUAL_PREDICTOR_H_INCLUDED

#include "GBPredictor.h"


class GBUsualPredictor: public GBPredictor {
public:
    GBUsualPredictor(const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const pytensor2* xTrain,
        const pytensor2* xValid,
        pytensorY* residuals,
        pytensorY* preds,
        pytensorY* validRes,
        pytensorY* validPreds);

    GBUsualPredictor(const Lab_t zeroPredictor,
        const TreeHolder& treeHolder,
        const size_t featureCnt);

    virtual ~GBUsualPredictor();

    pytensorY predict2d(const pytensor2& x) override;
    
    void predictTreeTrain(const size_t treeNum) override;
};

#endif // GBUSUAL_PREDICTOR_H_INCLUDED
