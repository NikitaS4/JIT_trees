#include "ArrayAdapter.h"


namespace Adapter {
    std::vector<Lab_t> ArrayAdapter::labelsToVector(py::array array) {        
        py::buffer_info buf = array.request();

        // check dimensions
        if (buf.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        
        // TODO: add type checking
        double* ptr = static_cast<double*>(buf.ptr); // get C-style array
        size_t size = buf.size; // get array size

        std::vector<Lab_t> target(size, 0);
        for (size_t i = 0; i < size; ++i) {
            target[i] = Lab_t(ptr[i]);
        }

        return target;
    }

    
    std::vector<FVal_t> ArrayAdapter::featuresToVector(py::array array) {        
        py::buffer_info buf = array.request();

        // check dimensions
        if (buf.ndim != 1)
            throw std::runtime_error("Number of dimensions must be 1");
        
        // TODO: add type checking
        double* ptr = static_cast<double*>(buf.ptr); // get C-style array
        size_t size = buf.size; // get array size

        std::vector<FVal_t> target(size, 0);
        for (size_t i = 0; i < size; ++i) {
            target[i] = FVal_t(ptr[i]);
        }

        return target;
    }

    
    std::vector<std::vector<FVal_t>> ArrayAdapter::featuresToMtx(py::array array) {
        py::buffer_info buf = array.request();

        // check dimensions
        if (buf.ndim != 2)
            throw std::runtime_error("Number of dimensions must be 2");

        // TODO: add type checking
        double* ptr = static_cast<double*>(buf.ptr); // get C-style array
        // array has shape (n, m)
        size_t n = buf.shape[0];
        size_t m = buf.shape[1];
        
        if (n * m == 0)
            throw std::runtime_error("Shapes must be greater 0");
        
        // init target vector with correct shapes
        std::vector<std::vector<FVal_t>> target(n, std::vector<double>(m, 0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                target[i][j] = ptr[i * n + j];
            }
        }
        
        return target;
    }


    py::array ArrayAdapter::labelsToPy(const std::vector<Lab_t>& array) {
        py::array target = py::cast(array);
        return target;
    }

    
    py::array ArrayAdapter::featuresToPy(const std::vector<FVal_t>& array) {
        py::array target = py::cast(array);
        return target;
    }


    py::array ArrayAdapter::featureMtxToPy(const std::vector<std::vector<FVal_t>>& array) {
        py::array target = py::cast(array);
        return target;
    }
};