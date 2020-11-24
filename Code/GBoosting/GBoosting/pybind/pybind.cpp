#include <pybind11/pybind11.h>

#include "../common/GBoosting.h"
#include "../common/TestLauncher.h"
#include "../common/History.h"


namespace py = pybind11;

PYBIND11_MODULE(JITtrees, m) {
    py::class_<GradientBoosting>(m, "GradientBoosting")
        .def(py::init<const size_t, const size_t>())
        .def("fit", &GradientBoosting::fit, "Fit regression model", py::arg("x_train"),
			py::arg("y_train"), py::arg("x_valid"),
			py::arg("y_valid"), py::arg("tree_count"),
			py::arg("tree_depth"), py::arg("learning_rate"))
        .def("predict", &GradientBoosting::predict)
        .def("printModel", &GradientBoosting::printModel);

    py::class_<TestLauncher>(m, "TestLauncher")
        .def(py::init<const std::vector<std::vector<FVal_t>>&,
        const std::vector<Lab_t>&,
        const std::vector<std::vector<FVal_t>>&,
        const std::vector<Lab_t>&> ())
        .def(py::init<const std::string&, const std::string&,
        const std::string&, const std::string&>())
        .def("performTest", &TestLauncher::performTest)
        .def("singleTestPrint", &TestLauncher::singleTestPrint);

    py::class_<History>(m, "History")
        .def(py::init<>())
        .def(py::init<const size_t, const std::vector<Lab_t>&,
        const std::vector<Lab_t>&> ())
        .def("addLosses", &History::addLosses)
        .def("addAllLosses", &History::addAllLosses)
        .def("setTreesLearnt", &History::setTreesLearnt)
        .def("getTreesLearnt", &History::getTreesLearnt)
        .def("getTrainLosses", &History::getTrainLosses)
        .def("getValidLosses", &History::getValidLosses);
}

