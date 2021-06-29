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


PYBIND11_MODULE(regbm, m) {
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
        .def(py::init<const size_t, const size_t, const size_t,
             const bool, const size_t>(), 
            "Gradient boosting model constructor",
            py::arg("min_bins")=dp::binsMin, 
            py::arg("max_bins")=dp::binsMax,
            py::arg("patience")=dp::patience,
            py::arg("no_early_stopping")=dp::noEs,
            py::arg("thread_cnt")=dp::threadCnt)
        .def(py::init<const std::string&, const size_t>(),
            "Load GB model from the file",
            py::arg("filename"),
            py::arg("thread_cnt")=dp::threadCnt)
        .def("fit", &GradientBoosting::fit, "Fit regression model", py::arg("x_train"),
            py::arg("y_train"), py::arg("x_valid"), py::arg("y_valid"),
            py::arg("tree_count")=dp::treeCount, 
            py::arg("tree_depth")=dp::treeDepth,
            py::arg("feature_fold_size")=dp::featureFoldSize,
            py::arg("learning_rate")=dp::learningRate,
            py::arg("regularization_param")=dp::regParam,
            py::arg("early_stopping_delta")=dp::earlyStoppingDelta,
            py::arg("batch_part")=dp::batchPart,
            py::arg("random_state")=dp::randomState,
            py::arg("batch_strategy")=dp::batchStrategy,
            py::arg("random_hist_thresholds")=dp::randThresholds,
            py::arg("remove_regularization_later")=dp::removeReg,
            py::arg("spoil_split_scores")=dp::spoilScores)
        .def("predict", static_cast<Lab_t (GradientBoosting::*)(const pytensor1&)const>(&GradientBoosting::predict), "Predict labels for a single sample",
            py::arg("x_test"))
        .def("predict", static_cast<pytensorY (GradientBoosting::*)(const pytensor2&)const>(&GradientBoosting::predict), "Predict labels for batch",
            py::arg("x_test"))
        .def("predict_from_to", &GradientBoosting::predictFromTo, "Predict labels for sample on a subset of trees",
            py::arg("x_test"), py::arg("from"), py::arg("to"))
        .def("save_model", static_cast<void (GradientBoosting::*)(const std::string&)const>(&GradientBoosting::saveModel), "Save GB model to the file",
            py::arg("filename"));
}
