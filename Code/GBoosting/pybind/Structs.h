#ifndef ADAPTERSTRUCTS_H
#define ADAPTERSTRUCTS_H

#include "../common/Structs.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py = pybind11;

// define dtype of numpy array
using pyarray = py::array_t<FVal_t, py::array::c_style | py::array::forcecast>;
using pyarrayY = py::array_t<Lab_t, py::array::c_style | py::array::forcecast>;


#endif // ADAPTERSTRUCTS_H