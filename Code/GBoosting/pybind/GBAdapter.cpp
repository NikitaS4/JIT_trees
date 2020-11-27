#include "GBAdapter.h"
#include "ArrayAdapter.h"


namespace Adapter {
    GradientBoosting::GradientBoosting(const size_t binCount,
    const size_t patience): boosting(binCount, patience) {}

    History GradientBoosting::fit(pyarray xTrain, pyarrayY yTrain, pyarray xValid,
        pyarrayY yValid, const size_t treeCount, const size_t treeDepth,
        const float learningRate) {
            // convert numpy arrays to vectors
            std::vector<std::vector<FVal_t>> xTrainVec = ArrayAdapter::featuresToMtx(xTrain);            
            std::vector<Lab_t> yTrainVec = ArrayAdapter::labelsToVector(yTrain);            
            std::vector<std::vector<FVal_t>> xValidVec = ArrayAdapter::featuresToMtx(xValid);
            std::vector<Lab_t> yValidVec = ArrayAdapter::labelsToVector(yValid);

            // run c++ implementation
            // use rvalued history to create history adapter to return to Python
            History history(boosting.fit(xTrainVec, yTrainVec, xValidVec, yValidVec, treeCount, treeDepth,
            learningRate));
            
            return history;
        }

    Lab_t GradientBoosting::predict(pyarray xTest) const {
        // convert numpy array to std::vector
        std::vector<FVal_t> xTestVec = ArrayAdapter::featuresToVector(xTest);
        // run c++ implementation
        return boosting.predict(xTestVec);
    }

    GradientBoosting::~GradientBoosting() {
        // dtor
    }
};