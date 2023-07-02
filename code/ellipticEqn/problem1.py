"""
problem 1, Dirichlet BC

domain is [-1, 1] x [-1, 1]
- div(grad(u)) = - y * (1 - y) * (1 - x - x^2/2) * exp(x+y)
                 - x * (1 - x/2) * (-3 * y - y^2) * exp(x+y)
u = -1.5 * y * (1-y) * exp(-1+y)  on x=-1
u = 0.5 * y * (1-y) * exp(1+y)    on x=1
u = -2 * x * (1-x/2) * exp(x-1)  on y=-1
u = 0  on y=1

u = x * y * (1 - x/2) * (1 - y) * exp(x + y)
"""


"""
For the general 2-D elliptic eqn, we have
- div(c grad(u)) = f

As for the problem 1, c = 1, and 
f(x,y) = - y * (1 - y) * (1 - x - x^2/2) * exp(x+y)
         - x * (1 - x/2) * (-3 * y - y^2) * exp(x+y)

a_{ij} = integrate c * grad(phi_j) * grad(psi_i)  dV
       = integrate c * ( (partial_x(phi_j) * partial_x(psi_i)) + (partial_y(phi_j) * partial_y(psi_i)) )
where psi_i is the test function, and phi_j is the trial function

b_{i} = integrate f(x,y) * psi_i dV
"""

import numpy as np
import sys
from mesh import TriMesh2d, BoundaryData2d
import fem2d
import scipy.sparse.linalg
import time
from numba import jit
np.set_printoptions(threshold=sys.maxsize)

def problem1(Nlb = 3):
    
    @jit(nopython=True, cache=True)
    def boundary_condition(x, y):
        tol = 1.e-14
        res = 0.
        if np.abs(x - (-1.)) < tol: # left
            res = -1.5 * y * (1.-y) * np.exp(-1.+y)
            return res
        elif np.abs(x - 1.) < tol:  # right
            res = 0.5 * y * (1.-y) * np.exp(1.+y)
            return res
        elif np.abs(y - (-1.)) < tol:  # bottom
            res = -2. * x * (1.-x/2.) * np.exp(x-1.)
            return res
        elif np.abs(y - 1.) < tol:  # upper
            res = 0.
            return res
        else:
            raise ValueError("point is not on the boundary.")

    @jit(nopython=True, cache=True)
    def exact_sol(x, y):
        res = x * y * (1. - x/2.) * (1. - y) * np.exp(x + y)
        return res

    time1 = time.time()

    # generate mesh grid, FE data, and BC data
    x1d_mesh_node = np.linspace(-1., 1., 257)
    y1d_mesh_node = np.linspace(-1., 1., 257)
    mesh2d = TriMesh2d(x1d_mesh_node, y1d_mesh_node, FE_node_num=Nlb)
    bc_data = BoundaryData2d(mesh2d.P, mesh2d.CT, mesh2d.Pb, mesh2d.CTb)

    time2 = time.time()
    print("generate mesh and BC data -> time usage = %f sec" % (time2 - time1))

    # assemble the matrix
    # cfunc_num = 1 corresponds to coeff_func(x,y) = 1
    matA = fem2d.get_matA(cfunc_num = 1, N = mesh2d.N, Nb = mesh2d.Nb,
                          matP = mesh2d.P, CmatT = mesh2d.CT, 
                          CmatTb_trial = mesh2d.CTb, \
                          CmatTb_test = mesh2d.CTb,  \
                          Nlb_trial = mesh2d.Nlb,  Nlb_test = mesh2d.Nlb, \
                          trial_func_ndx = 1,  trial_func_ndy = 0,  \
                          test_func_ndx = 1,   test_func_ndy = 0)

    time3 = time.time()
    print("generate 1st matA -> time usage = %f sec" % (time3 - time2))

    # print(matA.toarray())
    matA1 = fem2d.get_matA(cfunc_num = 1, N = mesh2d.N, Nb = mesh2d.Nb,
                          matP = mesh2d.P, CmatT = mesh2d.CT, 
                          CmatTb_trial = mesh2d.CTb, \
                          CmatTb_test = mesh2d.CTb,  \
                          Nlb_trial = mesh2d.Nlb,  Nlb_test = mesh2d.Nlb, \
                          trial_func_ndx = 0,  trial_func_ndy = 1,  \
                          test_func_ndx = 0,   test_func_ndy = 1)

    time4 = time.time()
    print("generate 2nd matA -> time usage = %f sec" % (time4 - time3))

    matA = (matA + matA1).tolil()

    time5 = time.time()
    print("add matA1 and matA2 -> time usage = %f sec" % (time5 - time4))

    # print(matA.toarray())
    # cfunc_num = 2 corresponds to coeff_func(x,y) = f(x,y) in problem 1
    vecb = fem2d.get_vecb (cfunc_num = 2, N = mesh2d.N, Nb = mesh2d.Nb,\
                           matP = mesh2d.P, CmatT = mesh2d.CT, \
                           CmatTb_test = mesh2d.CTb,  \
                           Nlb_test = mesh2d.Nlb, \
                           test_func_ndx = 0,  test_func_ndy = 0)

    time6 = time.time()
    print("generate vecb -> time usage = %f sec" % (time6 - time5))
    # print(np.asarray(vecb))

    # treate Dirichlet BC
    for i in range(bc_data.nbn):
        if bc_data.bcFEnode[0,i] == 0:  # dirichlet Bc
            temp_node_num = bc_data.bcFEnode[1,i]
            matA[temp_node_num, :] = 0.
            matA[temp_node_num, temp_node_num] = 1.
            vecb[temp_node_num] = boundary_condition(mesh2d.Pb[0,temp_node_num],\
                                                     mesh2d.Pb[1,temp_node_num])

    time7 = time.time()
    print("treate Dirichlet BC -> time usage = %f sec" % (time7 - time6))

    # solve A * x = b
    matA = matA.tocsr()
    vecx = scipy.sparse.linalg.spsolve(matA, vecb)


    time8 = time.time()
    print("solve Ax = b -> time usage = %f sec" % (time8 - time7))

    print("shape of matA", matA.shape)

    # estimate error
    err = np.zeros(mesh2d.Pb.shape[1])
    for i in range(err.shape[0]):
        err[i] = np.abs(vecx[i] - exact_sol(mesh2d.Pb[0,i], mesh2d.Pb[1,i]))
    print("*****" * 5)
    print("max abs err at all nodes:")
    print("%.5e" % (np.amax(err)))
    print("*****" * 5)

    time9 = time.time()
    print("estimate absolute error -> time usage = %f sec" % (time9 - time8))


    pass


if __name__ == '__main__':
    start_time = time.time()
    problem1(Nlb=6)
    end_time = time.time()
    print("total time usage = %f sec" % (end_time - start_time))
    pass
