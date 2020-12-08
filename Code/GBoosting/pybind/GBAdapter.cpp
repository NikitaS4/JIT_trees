#include "GBAdapter.h"
#include "ArrayAdapter.h"


namespace py = pybind11;


namespace Adapter {
    GradientBoosting::GradientBoosting(const size_t binCount,
    const size_t patience): boosting(binCount, patience) {}

    History GradientBoosting::fit(pyarray xTrain, pyarrayY yTrain, pyarray xValid,
        pyarrayY yValid, const size_t treeCount, const size_t treeDepth,
        const float learningRate, const Lab_t earlyStoppingDelta) {
            // convert numpy arrays to vectors
            std::vector<std::vector<FVal_t>> xTrainVec = ArrayAdapter::featuresToMtx(xTrain);            
            std::vector<Lab_t> yTrainVec = ArrayAdapter::labelsToVector(yTrain);            
            std::vector<std::vector<FVal_t>> xValidVec = ArrayAdapter::featuresToMtx(xValid);
            std::vector<Lab_t> yValidVec = ArrayAdapter::labelsToVector(yValid);

            // run c++ implementation
            // use rvalued history to create history adapter to return to Python
            History history(boosting.fit(xTrainVec, yTrainVec, xValidVec, yValidVec, treeCount, treeDepth,
            learningRate, earlyStoppingDelta));
            
            return history;
        }

    Lab_t GradientBoosting::predict(pyarray xTest) const {
        // convert numpy array to std::vector
        std::vector<FVal_t> xTestVec = ArrayAdapter::featuresToVector(xTest);
        // run c++ implementation
        return boosting.predict(xTestVec);
    }

    Lab_t GradientBoosting::predictFromTo(pyarray xTest, 
        const size_t firstEstimator, const size_t lastEstimator) const {
            // convert numpy array to std::vector
            std::vector<FVal_t> xTestVec = ArrayAdapter::featuresToVector(xTest);
            // run c++ implementation
            return boosting.predictFromTo(xTestVec, firstEstimator, lastEstimator);
        }

    GradientBoosting::~GradientBoosting() {
        // dtor
    }
};