"""
FEM basis functions and Gaussian quadrature
ymma98
"""

import cython


cdef double _affinemap1d_xhat_to_x(double x1, double x2, double xhat) nogil


cdef void _shape1d_node2(int idx, double x, double[:] node, double[:] phi) nogil

cdef void _shape1d_node3(int idx, double x, double[:] node, double[:] phi) nogil



cdef void _shape1d_node4(int idx, double x, double[:] node, double[:] phi) nogil

cdef void _gauss_legendre_quadrature_set1d(int n, \
                                           double[:] x, \
                                           double[:] w) nogil

cdef void _shape2d_t3(int idx, int idy,  \
                      double x, double y, \
                      double[:] node_x, double[:] node_y, double[:] phi) nogil



cdef void _shape2d_t6(int idx, int idy,  \
                      double x, double y, \
                      double[:] node_x, double[:] node_y, double[:] phi) nogil


cdef void _shape2d_t10(int idx, int idy,  \
                       double x, double y, \
                       double[:] node_x, double[:] node_y, double[:] phi) nogil

cdef void _gauss_quadrature_triangle(int n, \
                                    double[:] x, \
                                    double[:] y, \
                                    double[:] w) nogil
