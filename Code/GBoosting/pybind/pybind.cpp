#include <pybind11/pybind11.h>
#include "GBAdapter.h"
#include "HistoryAdapter.h"


namespace py = pybind11;

PYBIND11_MODULE(JITtrees, m) {
    py::class_<Adapter::History>(m, "History")
        .def("trees_number", &Adapter::History::getTreesLearnt,
        "Get the number of trees built during fit")
        .def("train_losses", &Adapter::History::getTrainLosses,
        "Get train losses array")
        .def("valid_losses", &Adapter::History::getValidLosses,
        "Get validation losses array");
    
    py::class_<Adapter::GradientBoosting>(m, "Boosting")
        .def(py::init<const size_t, const size_t>())
        .def("fit", &Adapter::GradientBoosting::fit, "Fit regression model", py::arg("x_train"),
            py::arg("y_train"), py::arg("x_valid"), py::arg("y_valid"),
            py::arg("tree_count"), py::arg("tree_depth"), py::arg("learning_rate"))
        .def("predict", &Adapter::GradientBoosting::predict, "Predict labels for sample",
            py::arg("x_test"));
    
};
