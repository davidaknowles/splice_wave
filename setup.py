# python setup.py build_ext --inplace

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("one_hot.pyx"),
    include_dirs=[np.get_include()]
)