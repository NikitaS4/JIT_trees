# JIT_trees

Gradient boosting implementation on c++ integrated with Python via [pybind11](https://github.com/pybind/pybind11) and [xtensor](https://github.com/xtensor-stack/xtensor-python).

Regression trees inside gradient boosting model are JIT-compiled. On Windows [mingw-w64](https://mingw-w64.org/doku.php/download/win-builds) must be installed and MinGw compiler must be visible to PATH. On Linux, GCC compiler is used for JIT-compilation.

# Module creation

Run the following in command prompt:

```
cd Code/GBoosting
python setup.py build_ext -i
```

Module will be created in the `Code/GBoosting` directory.

