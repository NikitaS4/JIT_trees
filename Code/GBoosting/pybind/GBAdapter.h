#ifndef GBADAPTER_H
#define GBADAPTER_H

#include "Structs.h"
#include "HistoryAdapter.h"
#include "../common/GBoosting.h"


namespace Adapter {
    class GradientBoosting {
    public:
        GradientBoosting(const size_t binCount, const size_t patience);
        virtual ~GradientBoosting();

        History fit(pyarray xTrain, pyarrayY yTrain, pyarray xValid,
        pyarrayY yValid, const size_t treeCount, const size_t treeDepth,
        const float learningRate, const Lab_t earlyStoppingDelta);

        Lab_t predict(pyarray xTest) const;
        Lab_t predictFromTo(pyarray xTest, 
        const size_t firstEstimator, const size_t lastEstimator) const;

    private:
        ::GradientBoosting boosting;
    };
};

#endif // GBADAPTER_H