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


cdef void _test():
    cdef int n = 4
    cdef double[3] a = [1.,2.,3.];
    print(a);
    cdef double[:] b = a;
    b[:] = 0.
    print(b[0], b[1], b[2]);
    cdef double[:] c = [1,2,3,4];
    print(c)

def test():
    _test()
