#ifndef GBADAPTER_H
#define GBADAPTER_H

#include "../common/GBoosting.h"
#include "HistoryAdapter.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py = pybind11;

namespace Adapter {
    class GradientBoosting {
    public:
        GradientBoosting(const size_t binCount, const size_t patience);
        virtual ~GradientBoosting();

        History fit(py::array xTrain, py::array yTrain, py::array xValid,
        py::array yValid, const size_t treeCount, const size_t treeDepth,
        const float learningRate);

        Lab_t predict(py::array xTest) const;

    private:
        ::GradientBoosting boosting;
    };
};

#endif // GBADAPTER_H