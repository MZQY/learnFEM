"""
FEM basis functions and Gaussian quadrature for parallel running
ymma98
"""

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt
from libc.stdio cimport printf
from libc.stdlib cimport exit


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _shape2d_t3(int idx, int idy,  \
                      double x, double y, \
                      double[3] node_x, double[3] node_y, double[3] phi) nogil:
    """
    quadratic shape function on the reference triangle with vertices (0,0), (1,0) and (0,1)
    The values are derived using Mathematica in basis.nb.
     |
     1  3
     |  ..
     |  . .
     y  .  .
     |  .   .
     |  .    .
     0  1-----2
     |
     +--0--x--1-->
    :param idx: (int) derivative order of x for shape function
    :param idy: (int) derivative order of y for shape function
    :param x: (double) required coordinate
    :param y: (double) required coordinate
    :param node_x: (double[3]) coordinate x of nodes
    :param node_y: (double[3]) coordinate y of nodes
    :param phi: (double[3])  shape function after idx order derivative on x and idy order derivative on y
    """
    node_x[0] = 0.   ;  node_y[0] = 0.
    node_x[1] = 1.   ;  node_y[1] = 0.
    node_x[2] = 0.   ;  node_y[2] = 1.
    if ( (x+y-1)<=0. and x >= 0. and y >= 0.):
        if (idx == 0 and idy == 0):
            phi[0] = 1. - x - y
            phi[1] = x
            phi[2] = y
        elif (idx == 1 and idy == 0):
            phi[0] = -1.
            phi[1] = 1.
            phi[2] = 0.
        elif (idx == 0 and idy == 1):
            phi[0] = -1.
            phi[1] = 0.
            phi[2] = 1.
        else:
            phi[0] = 0.
            phi[1] = 0.
            phi[2] = 0.
    else:
        phi[0] = 0.
        phi[1] = 0.
        phi[2] = 0.



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _shape2d_t6(int idx, int idy,  \
                      double x, double y, \
                      double[6] node_x, double[6] node_y, double[6] phi) nogil:
    """
    quadratic shape function on the reference triangle with vertices (0,0), (1,0) and (0,1)
    The values are derived using Mathematica in basis.nb.
     |
     1  3
     |  ..
     |  . .
     y  5  4
     |  .   .
     |  .    .
     0  1--6--2
     |
     +--0--x--1-->
    :param idx: (int) derivative order of x for shape function
    :param idy: (int) derivative order of y for shape function
    :param x: (double) required coordinate
    :param y: (double) required coordinate
    :param node_x: (double[6]) coordinate x of nodes
    :param node_y: (double[6]) coordinate y of nodes
    :param phi: (double[6])  shape function after idx order derivative on x and idy order derivative on y
    """
    node_x[0] = 0.       ;  node_y[0] = 0.
    node_x[1] = 1.       ;  node_y[1] = 0.
    node_x[2] = 0.       ;  node_y[2] = 1.
    node_x[3] = 1. / 2.  ;  node_y[3] = 1. / 2.
    node_x[4] = 0.       ;  node_y[4] = 1. / 2.
    node_x[5] = 1. / 2.  ;  node_y[5] = 0.
    if ( (x+y-1) <= 0. and x >= 0. and y >= 0.):
        if (idx == 0 and idy == 0):
            phi[0] = 1. + 2. * x*x - 3. * y + 2. * y*y + x * (-3. + 4.* y)
            phi[1] = x * (-1. + 2. * x)
            phi[2] = y * (-1. + 2. * y)
            phi[3] = 4. * x * y
            phi[4] = -4. * y * (-1. + x + y)
            phi[5] = -4. * x * (-1. + x + y)
        elif (idx == 1 and idy == 0):
            phi[0] = -3. + 4. * x + 4. * y
            phi[1] = -1. + 4. * x
            phi[2] = 0.
            phi[3] = 4. * y
            phi[4] = -4. * y
            phi[5] = -4. * (-1. + 2. * x + y)
        elif (idx == 0 and idy == 1):
            phi[0] = -3. + 4. * x + 4. * y
            phi[1] = 0.
            phi[2] = -1. + 4. * y
            phi[3] = 4. * x
            phi[4] = -4. * (-1. + x + 2. * y)
            phi[5] = -4. * x
        elif (idx == 1 and idy == 1):
            phi[0] = 4.
            phi[1] = 0.
            phi[2] = 0.
            phi[3] = 4.
            phi[4] = -4.
            phi[5] = -4.
        elif (idx == 2 and idy == 0):
            phi[0] = 4.
            phi[1] = 4.
            phi[2] = 0.
            phi[3] = 0.
            phi[4] = 0.
            phi[5] = -8.
        elif (idx == 0 and idy == 2):
            phi[0] = 4.
            phi[1] = 0.
            phi[2] = 4.
            phi[3] = 0.
            phi[4] = -8.
            phi[5] = 0.
    else:
        phi[0] = 0.
        phi[1] = 0.
        phi[2] = 0.
        phi[3] = 0.
        phi[4] = 0.
        phi[5] = 0.




@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _shape2d_t10(int idx, int idy,  \
                       double x, double y, \
                       double[10] node_x, double[10] node_y, double[10] phi) nogil:
    """
    quadratic shape function on the reference triangle with vertices (0,0), (1,0) and (0,1)
    The values are derived using Mathematica in basis.nb.
     |
     1  3
     |  ..
     |  . .
     |  7  5
     y  .   .
     |  .    .
     |  6 10  4
     |  .      .
     0  1--8--9-2
     |
     +--0--x--1-->
    :param idx: (int) derivative order of x for shape function
    :param idy: (int) derivative order of y for shape function
    :param x: (double) required coordinate
    :param y: (double) required coordinate
    :param node_x: (double[10]) coordinate x of nodes
    :param node_y: (double[10]) coordinate y of nodes
    :param phi: (double[10])  shape function after idx order derivative on x and idy order derivative on y
    """
    node_x[0] = 0.       ;  node_y[0] = 0.
    node_x[1] = 1.       ;  node_y[1] = 0.
    node_x[2] = 0.       ;  node_y[2] = 1.
    node_x[3] = 2. / 3.  ;  node_y[3] = 1. / 3.
    node_x[4] = 1. / 3.  ;  node_y[4] = 2. / 3.
    node_x[5] = 0.       ;  node_y[5] = 1. / 3.
    node_x[6] = 0.       ;  node_y[6] = 2. / 3.
    node_x[7] = 1. / 3.  ;  node_y[7] = 0.
    node_x[8] = 2. / 3.  ;  node_y[8] = 0.
    node_x[9] = 1. / 3.  ;  node_y[9] = 1. / 3.
    if ( (x+y-1)<=0. and x >= 0. and y >=0.):
        if (idx == 0 and idy == 0):
            phi[0] = (2. - 9. * x*x*x - 11. * y + 18.* y*y - 9.* y*y*y - 9. * x*x *(-2. + 3. * y) + x * (-11. + 36. * y - 27. * y*y))/2.
            phi[1] = (x * (2. - 9. * x + 9. * x*x ))/2.
            phi[2] = (y * (2. - 9. * y + 9. * y*y ))/2.
            phi[3] = (9.* x * (-1. + 3. * x ) * y)/2.
            phi[4] = (9. * x * y * (-1. + 3. * y))/2.
            phi[5] = (9. * y * (2. + 3.* x*x - 5. * y + 3 * y*y + x * (-5. + 6 * y) ))/2.
            phi[6] = (-9. * y * (-1. + x + y) * (-1. + 3. * y))/2.
            phi[7] = (9. * x * (2. + 3. * x*x - 5. * y + 3. * y*y + x * (-5. + 6. * y)))/2.
            phi[8] = (-9. * x * (-1. + 3. * x) * (-1. + x + y))/2.
            phi[9] = -27. * x * y * (-1. + x + y)
        elif (idx == 1 and idy == 0):
            phi[0] = (-11. - 27. * x*x + x * (36. - 54. * y) + 36. * y - 27. * y*y)/2.
            phi[1] = 1. - 9. * x + (27. * x*x )/2.
            phi[2] = 0.
            phi[3] = (9. * (-1. + 6. * x) * y)/2.
            phi[4] = (9. * y * (-1. + 3. * y ))/2.
            phi[5] = (9. * y * (-5. + 6. * x + 6. * y))/2.
            phi[6] = (9. * (1. - 3. * y ) * y)/2.
            phi[7] = (9. * (2. + 9. * x*x - 5. * y + 3. * y*y + 2. * x * (-5. + 6. * y)))/2.
            phi[8] = (-9. * (1. + 9. * x*x - y + x * (-8. + 6. * y)))/2.
            phi[9] = -27. * y * (-1. + 2. * x + y)
        elif (idx == 0 and idy == 1):
            phi[0] = (-11. - 27. * x*x + x * (36. - 54. * y) + 36. * y - 27. * y*y)/2.
            phi[1] = 0.
            phi[2] = 1. - 9. * y + (27. * y*y)/2.
            phi[3] = (9. * x * (-1. + 3. * x ))/2.
            phi[4] = (9. * x * (-1. + 6. * y ))/2.
            phi[5] = (9. * (2. + 3. * x*x - 10. * y + 9. * y*y + x * (-5. + 12. * y)))/2.
            phi[6] = (-9. * (1. - 8. * y + 9. * y*y + x * (-1. + 6. * y)))/2.
            phi[7] = (9. * x * (-5. + 6. * x + 6. * y))/2.
            phi[8] = (9. * (1. - 3. * x) * x)/2.
            phi[9] = -27. * x * (-1. + x + 2. * y)
        elif (idx == 1 and idy == 1):
            phi[0] = -9. * (-2. + 3. * x + 3. * y)
            phi[1] = 0.
            phi[2] = 0.
            phi[3] = -4.5 + 27. * x
            phi[4] = -4.5 + 27. * y
            phi[5] = (9. * (-5. + 6. * x + 12. * y))/2.
            phi[6] = 4.5 - 27. * y
            phi[7] = (9. * (-5. + 12. * x + 6. * y))/2.
            phi[8] = 4.5 - 27. * x
            phi[9] = -27. * (-1. + 2. * x + 2. * y)
        elif (idx == 2 and idy == 0):
            phi[0] = -9. * (-2. + 3. * x + 3. * y)
            phi[1] = -9. + 27. * x
            phi[2] = 0.
            phi[3] = 27. * y
            phi[4] = 0.
            phi[5] = 27. * y
            phi[6] = 0.
            phi[7] = 9. * (-5. + 9. * x + 6. * y)
            phi[8] = -9. * (-4. + 9. * x + 3. * y)
            phi[9] = -54. * y
        elif (idx == 0 and idy == 2):
            phi[0] = -9. * (-2. + 3. * x + 3. * y)
            phi[1] = 0.
            phi[2] = -9. + 27. * y
            phi[3] = 0.
            phi[4] = 27. * x
            phi[5] = 9. * (-5. + 6. * x + 9. * y)
            phi[6] = -9. * (-4. + 3. * x + 9. * y)
            phi[7] = 27. * x
            phi[8] = 0.
            phi[9] = -54. * x
        elif (idx == 1 and idy == 2):
            phi[0] = -27.
            phi[1] = 0.
            phi[2] = 0.
            phi[3] = 0.
            phi[4] = 27.
            phi[5] = 54.
            phi[6] = -27.
            phi[7] = 27.
            phi[8] = 0.
            phi[9] = -54.
        elif (idx == 2 and idy == 1):
            phi[0] = -27.
            phi[1] = 0.
            phi[2] = 0.
            phi[3] = 27.
            phi[4] = 0.
            phi[5] = 27.
            phi[6] = 0.
            phi[7] = 54.
            phi[8] = -27.
            phi[9] = -54.
        elif (idx == 3 and idy == 0):
            phi[0] = -27.
            phi[1] = 27.
            phi[2] = 0.
            phi[3] = 0.
            phi[4] = 0.
            phi[5] = 0.
            phi[6] = 0.
            phi[7] = 81.
            phi[8] = -81.
            phi[9] = 0.
        elif (idx == 0 and idy == 3):
            phi[0] = -27.
            phi[1] = 0.
            phi[2] = 27.
            phi[3] = 0.
            phi[4] = 0.
            phi[5] = 81.
            phi[6] = -81.
            phi[7] = 0.
            phi[8] = 0.
            phi[9] = 0.
    else:
        phi[0] = 0.
        phi[1] = 0.
        phi[2] = 0.
        phi[3] = 0.
        phi[4] = 0.
        phi[5] = 0.
        phi[6] = 0.
        phi[7] = 0.
        phi[8] = 0.
        phi[9] = 0.


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _gauss_quad_triangle_6p(double[6] x, double[6] y, double[6] w) nogil:
    #  6 points, precision 4, Strang and Fix, formula #5
    x[0] = 0.816847572980459
    x[1] = 0.091576213509771
    x[2] = 0.091576213509771
    x[3] = 0.108103018168070
    x[4] = 0.445948490915965
    x[5] = 0.445948490915965

    y[0] = 0.091576213509771
    y[1] = 0.816847572980459
    y[2] = 0.091576213509771
    y[3] = 0.445948490915965
    y[4] = 0.108103018168070
    y[5] = 0.445948490915965

    w[0] = 0.109951743655322
    w[1] = 0.109951743655322
    w[2] = 0.109951743655322
    w[3] = 0.223381589678011
    w[4] = 0.223381589678011
    w[5] = 0.223381589678011

cdef void _gauss_quad_triangle_9p(double[9] x, double[9] y, double[9] w) nogil:
    # 9 points, precision 6, Strang and Fix formula #8.
    x[0] = 0.124949503233232
    x[1] = 0.437525248383384
    x[2] = 0.437525248383384
    x[3] = 0.797112651860071
    x[4] = 0.797112651860071
    x[5] = 0.165409927389841
    x[6] = 0.165409927389841
    x[7] = 0.037477420750088
    x[8] = 0.037477420750088

    y[0] = 0.437525248383384
    y[1] = 0.124949503233232
    y[2] = 0.437525248383384
    y[3] = 0.165409927389841
    y[4] = 0.037477420750088
    y[5] = 0.797112651860071
    y[6] = 0.037477420750088
    y[7] = 0.797112651860071
    y[8] = 0.165409927389841

    w[0] = 0.205950504760887
    w[1] = 0.205950504760887
    w[2] = 0.205950504760887
    w[3] = 0.063691414286223
    w[4] = 0.063691414286223
    w[5] = 0.063691414286223
    w[6] = 0.063691414286223
    w[7] = 0.063691414286223
    w[8] = 0.063691414286223



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _gauss_quadrature_triangle(int n, \
                                    double[:] x, \
                                    double[:] y,
                                    double[:] w) nogil:
    """
    get Gaussian quadrature points and weights for the reference triangle
      ^
    1 | *
      | ..
    Y | . .
      | .  .
    0 | *---*
      +------->
       0 X 1
    Integrate F(x, y) dX dY = Area(T) * Sum (1 <= I <= N) W(I) * F(X(I), Y(I)), 
    Area(T) = 1/2 here
    ref: https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
    
    :param n: (int) number of node points
    :param x: (double[n]) x coordinate of the quadrature point
    :param y: (double[n]) y coordinate of the quadrature point
    :param w: (double[n]) weights for Gauss-Legendre quadrature

    author: ymma98

    this function is modified from
        void triangle_unit_set ( int rule, double xtab[], double ytab[], double weight[] )
        which is created by John Burkardt

    references:
    H R Schwarz,
    Methode der Finiten Elemente,
    Teubner Studienbuecher, 1980.

    Strang and Fix,
    An Analysis of the Finite Element Method,
    Prentice Hall, 1973, page 184.

    Arthur H Stroud,
    Approximate Calculation of Multiple Integrals,
    Prentice Hall, 1971.

    O C Zienkiewicz,
    The Finite Element Method,
    McGraw Hill, Third Edition, 1977, page 201.

    """
    # 3 points, precision 2, Strang and Fix formula #2.
    if (n == 3):
        x[0] = 0.0
        x[1] = 1.0 / 2.0
        x[2] = 1.0 / 2.0

        y[0] = 1.0 / 2.0
        y[1] = 0.0
        y[2] = 1.0 / 2.0

        w[0] = 1.0 / 3.0
        w[1] = 1.0 / 3.0
        w[2] = 1.0 / 3.0
    # 4 points, precision 3, Strang and Fix formula #3
    elif (n == 4):
        x[0] = 10.0 / 30.0
        x[1] = 18.0 / 30.0
        x[2] =  6.0 / 30.0
        x[3] =  6.0 / 30.0

        y[0] = 10.0 / 30.0
        y[1] =  6.0 / 30.0
        y[2] = 18.0 / 30.0
        y[3] =  6.0 / 30.0

        w[0] = -27.0 / 48.0
        w[1] =  25.0 / 48.0
        w[2] =  25.0 / 48.0
        w[3] =  25.0 / 48.0
    #  6 points, precision 4, Strang and Fix, formula #5
    elif (n==6):
        x[0] = 0.816847572980459
        x[1] = 0.091576213509771
        x[2] = 0.091576213509771
        x[3] = 0.108103018168070
        x[4] = 0.445948490915965
        x[5] = 0.445948490915965

        y[0] = 0.091576213509771
        y[1] = 0.816847572980459
        y[2] = 0.091576213509771
        y[3] = 0.445948490915965
        y[4] = 0.108103018168070
        y[5] = 0.445948490915965

        w[0] = 0.109951743655322
        w[1] = 0.109951743655322
        w[2] = 0.109951743655322
        w[3] = 0.223381589678011
        w[4] = 0.223381589678011
        w[5] = 0.223381589678011
    # 9 points, precision 6, Strang and Fix formula #8.
    elif (n==9):
        x[0] = 0.124949503233232
        x[1] = 0.437525248383384
        x[2] = 0.437525248383384
        x[3] = 0.797112651860071
        x[4] = 0.797112651860071
        x[5] = 0.165409927389841
        x[6] = 0.165409927389841
        x[7] = 0.037477420750088
        x[8] = 0.037477420750088

        y[0] = 0.437525248383384
        y[1] = 0.124949503233232
        y[2] = 0.437525248383384
        y[3] = 0.165409927389841
        y[4] = 0.037477420750088
        y[5] = 0.797112651860071
        y[6] = 0.037477420750088
        y[7] = 0.797112651860071
        y[8] = 0.165409927389841

        w[0] = 0.205950504760887
        w[1] = 0.205950504760887
        w[2] = 0.205950504760887
        w[3] = 0.063691414286223
        w[4] = 0.063691414286223
        w[5] = 0.063691414286223
        w[6] = 0.063691414286223
        w[7] = 0.063691414286223
        w[8] = 0.063691414286223

    else:
        printf("******************************")
        printf("Illegal value of n = %d \n", n)
        printf("Legal values are 3, 4, 6, 9")
        printf("******************************")
        exit(1)
