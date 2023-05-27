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

cdef void _test2(double[:] a, double[:] b):
    cdef double[3] a1
    a1[:] = [5,6,7]
    cdef double[3] b2
    b2[:] = [7., 8, 9]
    a[:] = a1
    b[:] = b2
    
    
cdef void _test3(double[:] a, double[:] b):
    cdef double[3] a1
    a1[:] = [5,6,7]
    cdef double[3] b2
    b2[:] = [7., 8, 9]
    for i in range(a.shape[0]):
        if(a[i] == 1):
            raise ValueError("hahaha")
        a[i] = a1[i]
        b[i] = b2[i]

cdef void _test4(double[:] a, double[:] b) nogil:
    a[0] = 1.
    a[1] = 2.
    b[0] = 3.
    b[1] = 4.


cdef void _test():
    cdef int n = 4
    cdef double[3] a = [1.,2.,3.];
    print(a);
    cdef double[:] b = a;
    b[:] = 0.
    print(b[0], b[1], b[2]);
    print(a);
    # cdef double[:] c;
    # c = [1,2,3,4]
    # print(c)

def test():
    _test()

def test2(a, b):
    _test2(a, b)

def test3(a, b):
    _test3(a, b)

def test4(a, b):
    _test4(a, b)
