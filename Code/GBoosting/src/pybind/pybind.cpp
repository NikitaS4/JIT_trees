#include "pybind11/pybind11.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"

#include "../common/History.h"
#include "../common/GBoosting.h"
#include "defaultParameters.h"


namespace py = pybind11;
namespace dp = defaultParams;


PYBIND11_MODULE(JITtrees, m) {
    xt::import_numpy();
    
    m.doc() = "Gradient boosting implementation";

    py::class_<History>(m, "History")
        .def("trees_number", &History::getTreesLearnt,
        "Get the number of trees built during fit")
        .def("train_losses", &History::getTrainLosses,
        "Get train losses array")
        .def("valid_losses", &History::getValidLosses,
        "Get validation losses array");
    
    py::class_<GradientBoosting>(m, "Boosting")
        .def(py::init<const size_t, const size_t>(), "Gradient boosting model constructor",
            py::arg("bins")=dp::bins, py::arg("patience")=dp::patience)
        .def("fit", &GradientBoosting::fit, "Fit regression model", py::arg("x_train"),
            py::arg("y_train"), py::arg("x_valid"), py::arg("y_valid"),
            py::arg("tree_count")=dp::treeCount, 
            py::arg("tree_depth")=dp::treeDepth,
            py::arg("learning_rate")=dp::learningRate,
            py::arg("early_stopping_delta")=dp::earlyStoppingDelta,
            py::arg("JIT")=dp::useJIT,
            py::arg("JITedCodeType")=dp::JITedCodeType)
        .def("predict", static_cast<Lab_t (GradientBoosting::*)(const pytensor1&)const>(&GradientBoosting::predict), "Predict labels for a single sample",
            py::arg("x_test"))
        .def("predict", static_cast<pytensorY (GradientBoosting::*)(const pytensor2&)const>(&GradientBoosting::predict), "Predict labels for batch",
            py::arg("x_test"))
        .def("predict_from_to", &GradientBoosting::predictFromTo, "Predict labels for sample on a subset of trees",
            py::arg("x_test"), py::arg("from"), py::arg("to"));
    
}
