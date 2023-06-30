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


cdef double coeff_functions(int coeff_num, double x, double y) nogil:
    cdef double res = 0.
    if (coeff_num == 0):
        pass
    elif (coeff_num == 1):
        pass
    else:
        printf("******************************");
        printf("wrong coeff_num for coeff_functions");
        printf("******************************");
        exit(1)
    return res


