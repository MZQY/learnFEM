"""
FEM basis functions and Gaussian quadrature
ymma98
"""

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt
from libc.stdio cimport printf



# def test_func(x, y):
#     return _test_pass_func_to_cython(_add, x, y)
# 
# cdef double _add(double x, double y) nogil:
#     return (x+y)

ctypedef double (*COEFF_FUNC)(double x, double y) nogil

cdef double _test_pass_func_to_cython(COEFF_FUNC func, double x, double y) nogil:
    return func(x,y)

