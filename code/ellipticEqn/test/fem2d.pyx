"""
FEM basis functions and Gaussian quadrature
ymma98
"""

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt, fabs
from libc.stdio cimport printf
from libc.stdlib cimport exit


cdef double coeff_function(double x, double y):
    return 2.


# https://stackoverflow.com/questions/18348083/passing-cython-function-to-cython-function
ctypedef double (*COEFF_FUNC)(double x, double y) nogil

def test(COEFF_FUNC cfunc, double x, double y):
    return cfunc(x, y)


