"""
FEM basis functions and Gaussian quadrature
ymma98
"""

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt, fabs, exp
from libc.stdio cimport printf
from libc.stdlib cimport exit


cdef double coeff_functions(int coeff_num, double x, double y) nogil:
    cdef double res = 0.
    if (coeff_num == 0):
        pass
    elif (coeff_num == 1):
        res = 1.
    elif (coeff_num == 2):
        res = -y * (1 - y) * (1 - x - x*x/2.) * exp(x+y) - x * (1. - x/2.) * (-3. * y - y*y) * exp(x+y)
    else:
        printf("******************************");
        printf("wrong coeff_num for coeff_functions");
        printf("******************************");
        exit(1)
    return res


