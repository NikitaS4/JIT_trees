#ifndef ARRAYADAPTER_H
#define ARRAYADAPTER_H

#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../common/Structs.h"


namespace py = pybind11;

namespace Adapter {
    class ArrayAdapter {
    public:
        // numpy arrays to std::vectors
        static std::vector<Lab_t> labelsToVector(py::array array);
        static std::vector<FVal_t> featuresToVector(py::array array);
        static std::vector<std::vector<FVal_t>> featuresToMtx(py::array array);

        // std::vectors to numpy arrays
        static py::array labelsToPy(const std::vector<Lab_t>& array);
        static py::array featuresToPy(const std::vector<FVal_t>& array);
        static py::array featureMtxToPy(const std::vector<std::vector<FVal_t>>& array);
    private:
        ArrayAdapter() {};
        ~ArrayAdapter() {};
    };
};

#endif // ARRAYADAPTER_H
