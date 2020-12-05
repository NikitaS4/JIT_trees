from glob import glob
from setuptools import setup
import pybind11
from pybind11.setup_helpers import Pybind11Extension


ext_modules = [
    Pybind11Extension(
        "JITtrees",
        sorted(glob("common/*.cpp")) + sorted(glob("pybind/*.cpp")),
        include_dirs=[pybind11.get_include(), "common", "pybind"],
        language="c++",
        extra_compile_args=['-std=c++17']
    ),
]


setup(
    name="JITtrees",
    ext_modules=ext_modules,
    requires=['pybind11']
)
