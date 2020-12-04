#ifndef ADAPTERSTRUCTS_H
#define ADAPTERSTRUCTS_H

#include "../common/Structs.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


// define dtype of numpy array
using pyarray = pybind11::array_t<FVal_t, pybind11::array::c_style | pybind11::array::forcecast>;
using pyarrayY = pybind11::array_t<Lab_t, pybind11::array::c_style | pybind11::array::forcecast>;


#endif // ADAPTERSTRUCTS_H