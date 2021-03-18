# JIT_trees

Gradient boosting (for regression task) implementation on c++ integrated with Python via [pybind11](https://github.com/pybind/pybind11) and [xtensor](https://github.com/xtensor-stack/xtensor-python).

Regression trees inside gradient boosting model can be precompiled into `*.so` or `*.DLL` files. On Windows to use precompiled trees, [mingw-w64](https://mingw-w64.org/doku.php/download/win-builds) must be installed and MinGw compiler must be visible to PATH. On Linux, GCC compiler is used for precompilation of the trees.


# Module creation

To create module, You should have CMake installed.

Before creating module, put the correct paths to `Code/GBoosting/CMakeLists.txt` to include *numpy* and *xtensor-python* directories.

Run the following in command prompt:

```
cd Code/GBoosting
python setup.py build_ext -i
```

Module will be created in the `Code/GBoosting` directory.


# Tests and examples

To run tests, open `tests` directory:

```
cd Code/GBoosting/tests
```

Firstly, You can run the `testLauncher.py` program to ensure the module was created correctly:

```
python testLauncher.py --all-fast
```

You can pass `--help` argument to see all options:

```
python testLauncher.py --help
```

For example, You can run tests in verbose mode and compare results to the Scikit-learn Gradient boosting:

```
python testLauncher.py -v 3 -s
```

Or You can make plots of losses and predictions:

```
python testLauncher.py -v 3 -p
```

To fit and test the model on the real data, You can run Jupyter Notebook:

```
jupyter notebook
```

And open `TestOnRealData.ipynb` notebook in the Jupyter Notebook. Datasets used in the notebook requires *scikit-learn* library to be installed. There's also an extra dataset Superconductivity that can be downloaded from [here](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data).


# Improvements

1. Gradient boosting is based on histograms - decision trees are built with thresholds got as the borders of the buckets of the histograms

2. Stochastic gradient boosting - each tree can be grown on a subset of the whole data (batch). There can be used a subset of features to build each tree also.

3. Threshold for each split is taken as a random number in the interval from the left border of the left bucket of the histogram to the right border of the right bucket of the histogram

4. Early stopping. When validation loss starts to grow, the fit process stops.

5. "Weak" trees are made weaker. Score of each split is spoiled with the random value, this helps to choose not optimal split and tends to diminish overvitting. Furthermore, histograms are created with the dynamic number of the buckets which also helps to fight overfitting.
