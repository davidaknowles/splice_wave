# python setup.py build_ext --inplace

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("utils/one_hot.pyx", "utils/get_fasta.py"),
    include_dirs=[np.get_include()]
)