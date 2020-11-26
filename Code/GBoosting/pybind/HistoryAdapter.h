#ifndef HISTORYADAPTER_H
#define HISTORYADAPTER_H

#include "../common/History.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py = pybind11;


namespace Adapter {
    class History {
    public:
    History(::History&& history);

    // getters only
    size_t getTreesLearnt();
    py::array getTrainLosses();
    py::array getValidLosses();
    private:
    ::History history;
    };
};

#endif
