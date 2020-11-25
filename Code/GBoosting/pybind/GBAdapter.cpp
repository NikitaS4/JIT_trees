#include "GBAdapter.h"
#include "ArrayAdapter.h"


namespace Adapter {
    GradientBoosting::GradientBoosting(const size_t binCount,
    const size_t patience): boosting(binCount, patience) {}

    void GradientBoosting::fit(py::array xTrain, py::array yTrain, py::array xValid,
        py::array yValid, const size_t treeCount, const size_t treeDepth,
        const float learningRate) {
            // convert numpy arrays to vectors
            std::vector<std::vector<FVal_t>> xTrainVec = ArrayAdapter::featuresToMtx(xTrain);
            std::vector<Lab_t> yTrainVec = ArrayAdapter::labelsToVector(yTrain);
            std::vector<std::vector<FVal_t>> xValidVec = ArrayAdapter::featuresToMtx(xValid);
            std::vector<Lab_t> yValidVec = ArrayAdapter::labelsToVector(yValid);

            // run c++ implementation
            History history = boosting.fit(xTrainVec, yTrainVec, xValidVec, yValidVec, treeCount, treeDepth,
            learningRate);
            // TODO: return history
        }

    Lab_t GradientBoosting::predict(py::array xTest) const {
        // convert numpy array to std::vector
        std::vector<FVal_t> xTestVec = ArrayAdapter::featuresToVector(xTest);
        // run c++ implementation
        return boosting.predict(xTestVec);
    }

    GradientBoosting::~GradientBoosting() {
        // dtor
    }
};