from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(Extension(
    'fem2d',
    sources=['fem2d.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    extra_compile_args = ["-O3", '-fopenmp'],
    extra_link_args = ['-fopenmp'],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
)))

