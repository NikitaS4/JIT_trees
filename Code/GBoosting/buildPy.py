from glob import glob
from setuptools import setup
from setuptools.command.build_ext import build_ext
import pybind11
from pybind11.setup_helpers import Pybind11Extension


# running this script with 'build_ext' option will cause warning
# this is because compilation is forced to run with -Wstrict-prototypes
# flag which is not used in c++ compilation
# use the class below to avoid warnings
class Build_ext_wrapper(build_ext):
    def build_extensions(self):
        try:
            if '-Wstrict-prototypes' in self.compiler.compiler_so:
                self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except:
            # But on Windows with another compiler there still can be problems
            pass
        finally:
            super().build_extensions()


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
    author="Schastlivtsev Nikita",
    author_email="schastlivtsevn@gmail.com",
    ext_modules=ext_modules,
    requires=['pybind11'],
    cmdclass={'build_ext': Build_ext_wrapper}
)
