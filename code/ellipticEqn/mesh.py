import numpy as np
import matplotlib.pyplot as plt
import basis


class Mesh1d:
    def __init__(self, x1d_nodes, FE_node_num, print_flag=False):
        """
        x1d_nodes: (np.ndarray, 1d) defines 1d nodes
        FE_node_num: (int) determines the node number in a FE element, which \
                    also determines the basis type. 2 for linear basis, 3 for \
                    quadratic basis and so on.
        """
        self.xmin = np.amin(x1d_nodes)
        self.xmax = np.amax(x1d_nodes)
        self.Nm = x1d_nodes.shape[0]  # nodes number of mesh
        self.N = self.Nm - 1  # element number
        self.Nl = 2  # node number on a single mesh element (segment line here)
        self.Nlb = FE_node_num  # number of FE nodes on a mesh element, which \
                                # is also the basis function number on a mesh element
        self.Nb = self.Nm + (self.Nlb - 2) * self.N  # number of total FE basis on mesh
                                # mesh point number + internal point number
        # mesh coordinate matrix
        self.P = x1d_nodes
        # mesh elment node number matrix, starts from 1
        self.T = np.arange(x1d_nodes.shape[0], dtype=np.int32) + 1
        self.CT = self.T - 1  # T matrix in C format (index starts from 0)
        # FE node coordinate matrix
        self.Pb = self._get_Pb()
        # FE node number matrix
        self.Tb = self._get_Tb()
        self.CTb = self.Tb - 1
        if print_flag:
            self._print()

    def _get_Pb(self):
        Pb = np.zeros(self.Nb)
        ref_nodes = np.zeros(self.Nlb);  temp_arr = np.zeros(self.Nlb)
        if self.Nlb == 2:
            return self.P
        elif self.Nlb == 3:
            basis.shape1d_node3(idx=0, x=0., node=ref_nodes, phi=temp_arr)
        elif self.Nlb == 4:
            basis.shape1d_node4(idx=0, x=0., node=ref_nodes, phi=temp_arr)
            pass
        else:
            raise ValueError("only FE_node_num=2 or 3 or 4 are available now")

        for i in range(self.N):
            x1 = self.P[i]
            x2 = self.P[i+1]
            loc_nodes = np.zeros(self.Nlb)
            for k in range(loc_nodes.shape[0]):
                loc_nodes[k] = basis.affinemap1d_xhat_to_x(x1, x2, ref_nodes[k])
            for j in range(self.Nlb):
                    Pb[i * self.Nlb - i + j] = loc_nodes[j]
        return Pb

    def _get_Tb(self):
        Tb = np.zeros((self.Nlb, self.N), dtype=np.int32)
        for n in range(self.N):
            for i in range(self.Nlb):
                Tb[i, n] = self.Nlb * n - n + i
        Tb += 1
        return Tb

    def _print(self):
        print("*****" * 4)
        print("mesh grid from %f to %f" % (self.xmin, self.xmax))
        print("T matrix is ")
        print(self.T)
        print("Tb matrix is ")
        print(self.Tb)
        print("P matrix is")
        print(self.P)
        print("Pb matrix is")
        print(self.Pb)
        print("*****" * 4)


class TriMesh2d:
    def __init__(self, x1d_nodes, y1d_nodes, FE_node_num, print_flag=False):
        self.xmin = np.amin(x1d_nodes)
        self.xmax = np.amax(x1d_nodes)
        self.ymin = np.amin(y1d_nodes)
        self.ymax = np.amax(y1d_nodes)
        self.x1d_mesh = x1d_nodes
        self.y1d_mesh = y1d_nodes
        self.nx = x1d_nodes.shape[0] - 1
        self.ny = y1d_nodes.shape[0] - 1
        self.N = self.nx * self.ny * 2
        self.Nm = (self.nx + 1) * (self.ny + 1)  # nodes number of mesh
        self.Nl = 3  # local mesh node number, 3 for triangle here
        self.Nlb = FE_node_num
        if self.Nlb == 3:
            self.Nb = self.Nm
            self.nbx = self.nx + 1  # basis node number in x dir
            self.nby = self.ny + 1  # basis node number in y dir
        elif self.Nlb == 6:
            self.Nb = (self.nx + 1 + self.nx) * (self.ny + 1 + self.ny)
            self.nbx = self.nx * 2 + 1
            self.nby = self.ny * 2 + 1
        elif self.Nlb == 10:
            self.Nb = (self.nx + 1 + self.nx * 2) * (self.ny + 1 + self.ny * 2)
            self.nbx = self.nx * 3 + 1
            self.nby = self.ny * 3 + 1
        else:
            raise ValueError("Nlb can only be 3, 6, 10")
        self.P = self._get_P()
        self.T = self._get_T()
        self.CT = self.T - 1
        self.Pb = self._get_Pb()
        self.Tb = self._get_Tb()
        self.CTb = self.Tb - 1
        if print_flag:
            self._print()

    def _get_P(self):
        """
        ^   4 --- 8 ---12 ---16 ---20
        |   |     |     |     |     |
        |   |     |     |     |     |
        |   3 --- 7 ---11 ---15 ---19
        |   |     |     |     |     |
        y   |     |     |     |     |
        |   2 --- 6 ---10 ---14 ---18
        |   |     |     |     |     |
        |   |     |     |     |     |
        |   1 --- 5 --- 9 --- 13 ---17

        |---------- x ------------- >
        """
        # self.nx = x1d_mesh.shape[0] - 1
        # self.ny = y1d_mesh.shape[0] - 1
        P = np.zeros((2, self.Nm))
        for i in range(self.x1d_mesh.shape[0]):
            for j in range(self.y1d_mesh.shape[0]):
                P[0, i * (self.ny + 1) + j] = self.x1d_mesh[i]
                P[1, i * (self.ny + 1) + j] = self.y1d_mesh[j]
        return P

    def _get_T(self):
        """
        ^   4--8--12
        |   |\ |\ |   
        |   | \| \|   
        |   3--7--11
        |   |\ |\ |   
        y   | \| \|   
        |   2--6--10
        |   |\ |\ |   
        |   | \| \|  
        |   1--5--9

        |---- x ---- >

        3       1 - 3
        |\       \  |
        | \       \ |
        |  \       \|
        1 --2       2
         """
        # self.nx = x1d_mesh.shape[0] - 1
        # self.ny = y1d_mesh.shape[0] - 1
        T = np.zeros((3, self.N), dtype=np.int32)
        for i in range(self.nx):
            for j in range(self.ny):
                """
                3--4
                |\ |
                | \|
                1 -2
                """
                p1 = i * (self.ny + 1) + j
                p3 = p1 + 1
                p2 = (i+1) * (self.ny + 1) + j
                p4 = p2 + 1
                T[0, i * 2 * self.ny + j * 2] = p1
                T[1, i * 2 * self.ny + j * 2] = p2
                T[2, i * 2 * self.ny + j * 2] = p3
                T[0, i * 2 * self.ny + j * 2 + 1] = p3
                T[1, i * 2 * self.ny + j * 2 + 1] = p2
                T[2, i * 2 * self.ny + j * 2 + 1] = p4
        T += 1
        return T

    def _get_Pb(self):
        """
        ^   4 --- 8 ---12 ---16 ---20
        |   |     |     |     |     |
        |   |     |     |     |     |
        |   3 --- 7 ---11 ---15 ---19
        |   |     |     |     |     |
        y   |     |     |     |     |
        |   2 --- 6 ---10 ---14 ---18
        |   |     |     |     |     |
        |   |     |     |     |     |
        |   1 --- 5 --- 9 --- 13 ---17

        |---------- x ------------- >
        """
        Pb = np.zeros((2, self.Nb))
        if (self.Nlb == 3):
            return self.P
        elif (self.Nlb == 6):
            # node number =       mesh node number       + inner node number
            x1d_FE = np.zeros(int(self.x1d_mesh.shape[0] + self.x1d_mesh.shape[0] - 1))
            y1d_FE = np.zeros(int(self.y1d_mesh.shape[0] + self.y1d_mesh.shape[0] - 1))
            for i in range(x1d_FE.shape[0]):
                if i%2 == 0:
                    # mesh node
                    x1d_FE[i] = self.x1d_mesh[int(i/2)]
            for i in range(x1d_FE.shape[0]):
                if i%2 == 1:
                    # inner FE node
                    x1d_FE[i] = (x1d_FE[i-1] + x1d_FE[i+1]) / 2.

            for i in range(y1d_FE.shape[0]):
                if i%2 == 0:
                    # mesh node
                    y1d_FE[i] = self.y1d_mesh[int(i/2)]
            for i in range(y1d_FE.shape[0]):
                if i%2 == 1:
                    # inner FE node
                    y1d_FE[i] = (y1d_FE[i-1] + y1d_FE[i+1]) / 2.
            
            for i in range(x1d_FE.shape[0]):
                for j in range(y1d_FE.shape[0]):
                    Pb[0, i * y1d_FE.shape[0] + j] = x1d_FE[i]
                    Pb[1, i * y1d_FE.shape[0] + j] = y1d_FE[j]
        elif (self.Nlb == 10):
            x1d_FE = np.zeros(int(self.x1d_mesh.shape[0] + (self.x1d_mesh.shape[0] - 1) * 2))
            y1d_FE = np.zeros(int(self.y1d_mesh.shape[0] + (self.y1d_mesh.shape[0] - 1) * 2))
            for i in range(x1d_FE.shape[0]):
                if i%3 == 0:
                    x1d_FE[i] = self.x1d_mesh[int(i/3)]
            for i in range(x1d_FE.shape[0]):
                if i%3 == 1:
                    x1d_FE[i] = x1d_FE[i-1] + (x1d_FE[i+2] - x1d_FE[i-1]) / 3.
                elif i%3 == 2:
                    x1d_FE[i] = x1d_FE[i-2] + (x1d_FE[i+1] - x1d_FE[i-2]) * 2. / 3.

            for i in range(y1d_FE.shape[0]):
                if i%3 == 0:
                    y1d_FE[i] = self.y1d_mesh[int(i/3)]
            for i in range(y1d_FE.shape[0]):
                if i%3 == 1:
                    y1d_FE[i] = y1d_FE[i-1] + (y1d_FE[i+2] - y1d_FE[i-1]) / 3.
                elif i%3 == 2:
                    y1d_FE[i] = y1d_FE[i-2] + (y1d_FE[i+1] - y1d_FE[i-2]) * 2. / 3.

            for i in range(x1d_FE.shape[0]):
                for j in range(y1d_FE.shape[0]):
                    Pb[0, i * y1d_FE.shape[0] + j] = x1d_FE[i]
                    Pb[1, i * y1d_FE.shape[0] + j] = y1d_FE[j]
        else:
            raise ValueError("Nlb can only be 3, 6, 10")
        
        return Pb


    def _get_Tb(self):
        Tb = np.zeros((self.Nlb, self.N), dtype=np.int32)
        if (self.Nlb == 3):
            return self.T
        elif (self.Nlb == 6):
            """
            p3 --- p6 --- p9
            |             |
            |             |
            p2     p5     p8
            |             |
            |             |
            p1 --- p4 --- p7
            """
            nby = self.ny + 1 + self.ny
            for i in range(self.nx):
                for j in range(self.ny):
                    p1 = i * 2 * nby + j * 2
                    p2 = p1 + 1
                    p3 = p1 + 2
                    p4 = p1 + nby
                    p5 = p4 + 1
                    p6 = p4 + 2
                    p7 = p1 + nby * 2
                    p8 = p7 + 1
                    p9 = p8 + 1
                    Tb[0, i * 2 * self.ny + j*2] = p1
                    Tb[1, i * 2 * self.ny + j*2] = p7
                    Tb[2, i * 2 * self.ny + j*2] = p3
                    Tb[3, i * 2 * self.ny + j*2] = p5
                    Tb[4, i * 2 * self.ny + j*2] = p2
                    Tb[5, i * 2 * self.ny + j*2] = p4

                    Tb[0, i * 2 * self.ny + j*2 + 1] = p3
                    Tb[1, i * 2 * self.ny + j*2 + 1] = p7
                    Tb[2, i * 2 * self.ny + j*2 + 1] = p9
                    Tb[3, i * 2 * self.ny + j*2 + 1] = p8
                    Tb[4, i * 2 * self.ny + j*2 + 1] = p6
                    Tb[5, i * 2 * self.ny + j*2 + 1] = p5
        elif (self.Nlb == 10):
            nby = self.ny + 1 + self.ny * 2
            for i in range(self.nx):
                for j in range(self.ny):
                    """
                    p4 --- p8 --- p12--- p16
                    |                    |
                    |                    |
                    p3     p7     p11    p15
                    |                    |
                    |                    | 
                    p2     p6     p10    p14
                    |                    |
                    |                    | 
                    p1 --- p5 --- p9 --- p13
                    """
                    p1 = i * 3 * nby + j * 3
                    p2 = p1 + 1
                    p3 = p1 + 2
                    p4 = p1 + 3
                    p5 = p1 + nby
                    p6 = p5 + 1
                    p7 = p5 + 2
                    p8 = p5 + 3
                    p9 = p5 + nby
                    p10 = p9 + 1
                    p11 = p9 + 2
                    p12 = p9 + 3
                    p13 = p9 + nby
                    p14 = p13 + 1
                    p15 = p13 + 2
                    p16 = p13 + 3
                    Tb[0, i * 2 * self.ny + j*2] = p1
                    Tb[1, i * 2 * self.ny + j*2] = p13
                    Tb[2, i * 2 * self.ny + j*2] = p4
                    Tb[3, i * 2 * self.ny + j*2] = p10
                    Tb[4, i * 2 * self.ny + j*2] = p7
                    Tb[5, i * 2 * self.ny + j*2] = p2
                    Tb[6, i * 2 * self.ny + j*2] = p3
                    Tb[7, i * 2 * self.ny + j*2] = p5
                    Tb[8, i * 2 * self.ny + j*2] = p9
                    Tb[9, i * 2 * self.ny + j*2] = p6

                    Tb[0, i * 2 * self.ny + j*2 + 1] = p4
                    Tb[1, i * 2 * self.ny + j*2 + 1] = p13
                    Tb[2, i * 2 * self.ny + j*2 + 1] = p16
                    Tb[3, i * 2 * self.ny + j*2 + 1] = p14
                    Tb[4, i * 2 * self.ny + j*2 + 1] = p15
                    Tb[5, i * 2 * self.ny + j*2 + 1] = p8
                    Tb[6, i * 2 * self.ny + j*2 + 1] = p12
                    Tb[7, i * 2 * self.ny + j*2 + 1] = p7
                    Tb[8, i * 2 * self.ny + j*2 + 1] = p10
                    Tb[9, i * 2 * self.ny + j*2 + 1] = p11
        else:
            raise ValueError("Nlb can only be 3, 6, 10")
        Tb += 1
        return Tb

    def _print(self):
        print("*****" * 4)
        print("mesh grid from %f to %f" % (self.xmin, self.xmax))
        print("T matrix is ")
        print(self.T)
        print("Tb matrix is ")
        print(self.Tb)
        print("P matrix is")
        print(self.P)
        print("Pb matrix is")
        print(self.Pb)
        print("*****" * 4)


class BoundaryData2d:
    """
    get bcMeshEdge matrix and bcFEnode matrix
    bcMeshEdge: mesh boundary edge matrix, only relates to mesh
    bcMeshEdge[0,k] : type of kth boundary edge
    bcMeshEdge[1,k] : index of element contains kth boundary edge
    bcMeshEdge[2,k] : global mesh node index of first end of kth BC edge
    bcMeshEdge[3,k] : global mesh node index of second end of kth BC edge

    bcFEnode : boundary FE node number matrix, only relates to FE space
    bcNode[0,k] : type of kth boundary node
    bcNode[1,k] : FE node index of kth FE node

    BC node type:
        0: dirichlet
        1: neumann
        2: robin
    """
    def __init__(self, Pmat, CTmat, Pbmat, CTbmat, print_flag=False):
        self.Nm = Pmat.shape[1]
        self.N = CTmat.shape[1]
        self.Nb = Pbmat.shape[1]
        self.Nlb = CTbmat.shape[0]
        # self.bcMeshEdge = np.zeros((4, self.nx*2+self.ny*2), dtype=np.int32)
        # self.bcFEnode = np.zeros((2, 2 * (self.nbx + self.nby) - 4), dtype=np.int32)
        self.xmax = np.amax(Pmat[0,:])
        self.xmin = np.amin(Pmat[0,:])
        self.ymax = np.amax(Pmat[1,:])
        self.ymin = np.amin(Pmat[1,:])
        self.bcMeshEdge = self._get_bcMeshEdge(Pmat, CTmat)
        self.bcFEnode = self._get_bcFEnode(Pbmat, CTbmat)

        if print_flag:
            self._print()

    def set_bctype(self, Pmat, Pbmat, bc_name, min, max, bctype):
        """
        :param bc_name: (string) 'top', 'bottom', 'right' or 'left'
        :param min: (double) min value in x dir or y dir
        :param max: (double) max value in x dir or y dir
        :param bctype: (int)
            0: dirichlet
            1: neumann
            2: robin
        """
        tol = 1.e-14
        for i in range(self.bcMeshEdge.shape[1]):
            p1_num = self.bcMeshEdge[2, i]
            p2_num = self.bcMeshEdge[3, i]
            p1_coord = Pmat[:, p1_num]
            p2_coord = Pmat[:, p2_num]
            x1 = p1_coord[0]
            y1 = p1_coord[1]
            x2 = p2_coord[0]
            y2 = p2_coord[1]

            if bc_name == 'top':
                if np.abs(y1-self.ymax) < tol and np.abs(y2-self.ymax) < tol:
                    if x1 <= max and x1 >= min and x2<= max and x2 >= min:
                        self.bcMeshEdge[0, i] = bctype
            elif bc_name == 'bottom':
                if np.abs(y1-self.ymin) < tol and np.abs(y2-self.ymin) < tol:
                    if x1 <= max and x1 >= min and x2<= max and x2 >= min:
                        self.bcMeshEdge[0, i] = bctype
            elif bc_name == 'right':
                if np.abs(x1-self.xmax) < tol and np.abs(x2-self.xmax) < tol:
                    if y1 <= max and y1 >= min and y2 <= max and y2 >= min:
                        self.bcMeshEdge[0,i] = bctype
            elif bc_name == 'left':
                if np.abs(x1-self.xmin) < tol and np.abs(x2-self.xmin) < tol:
                    if y1 <= max and y1 >= min and y2 <= max and y2 >= min:
                        self.bcMeshEdge[0,i] = bctype
            else:
                raise ValueError("wrong bc_name!")

        for i in range(self.bcFEnode.shape[1]):
            p_fe_num = self.bcFEnode[1, i]  # FE node number of a BC FE node point
            p_coord = Pbmat[:, p_fe_num]
            x = p_coord[0]
            y = p_coord[1]
            if bc_name == 'top':
                if np.abs(y-self.ymax) < tol:
                    if x <= max and x >= min:
                        self.bcFEnode[0, i] = bctype
            elif bc_name == 'bottom':
                if np.abs(y-self.ymin) < tol:
                    if x <= max and x >= min:
                        self.bcFEnode[0, i] = bctype
            elif bc_name == 'right':
                if np.abs(x-self.xmax) < tol:
                    if y <= max and y >= min:
                        self.bcFEnode[0, i] = bctype
            elif bc_name == 'left':
                if np.abs(x-self.xmin) < tol:
                    if y <= max and y >= min:
                        self.bcFEnode[0, i] = bctype
            else:
                raise ValueError("wrong bc_name!")



    def _get_bcMeshEdge(self, Pmat, CTmat):
        tol = 1.e-14
        bcMeshEdge = np.zeros((4,1), dtype=np.int32)
        for n in range(self.N):
            e_vertnum = CTmat[:,n]
            p1 = Pmat[:, e_vertnum[0]]  # coord
            p2 = Pmat[:, e_vertnum[1]]
            p3 = Pmat[:, e_vertnum[2]]
            # 3 cases: (p1, p2), (p1, p3), (p2, p3)
            if self._two_point_on_bc(p1, p2, tol=tol):
                temp = np.zeros((4,1), dtype=np.int32)
                temp[1,0] = n  # element number
                temp[2,0] = e_vertnum[0]
                temp[3,0] = e_vertnum[1] 
                bcMeshEdge = np.append(bcMeshEdge, temp, axis=1)
            if self._two_point_on_bc(p1, p3, tol=tol):
                temp = np.zeros((4,1), dtype=np.int32)
                temp[1,0] = n
                temp[2,0] = e_vertnum[0]
                temp[3,0] = e_vertnum[2]
                bcMeshEdge = np.append(bcMeshEdge, temp, axis=1)
            if self._two_point_on_bc(p2, p3, tol=tol):
                temp = np.zeros((4,1), dtype=np.int32)
                temp[1,0] = n
                temp[2,0] = e_vertnum[1]
                temp[3,0] = e_vertnum[2]
                bcMeshEdge = np.append(bcMeshEdge, temp, axis=1)
        bcMeshEdge = np.delete(bcMeshEdge, 0, axis=1)
        return bcMeshEdge

    def _get_bcFEnode(self, Pbmat, CTbmat):
        bcFEnode = np.zeros((2,1), dtype=np.int32)
        for n in range(self.N):
            FE_nodenumber = CTbmat[:,n] 
            for i in range(FE_nodenumber.shape[0]):
                if self._point_on_bc(Pbmat[:, FE_nodenumber[i]]):
                    temp = np.zeros((2,1), dtype=np.int32)
                    temp[0,0] = 0
                    temp[1,0] = FE_nodenumber[i]
                    bcFEnode = np.append(bcFEnode, temp, axis=1)
        bcFEnode = np.delete(bcFEnode, 0, axis=1)
        bcFEnode = np.unique(bcFEnode, axis=1)
        return bcFEnode


    def _two_point_on_bc(self, p1, p2, tol=1.e-14):
        """
        :param p1: (np.1darray) coord of a point p1[0] = x1, p1[1] = y1
        :param p2: (np.1darray) coord of a point p2[0] = x2, p1[1] = y2
        """
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        # xmin BC, a vertical line
        if np.abs(x1 - self.xmin) < tol and np.abs(x2 - self.xmin) < tol:
            return True
        elif np.abs(x1 - self.xmax) < tol and np.abs(x2 - self.xmax) < tol:
            return True
        elif np.abs(y1 - self.ymin) < tol and np.abs(y2 - self.ymin) < tol:
            return True
        elif np.abs(y1 - self.ymax) < tol and np.abs(y2 - self.ymax) < tol:
            return True
        else:
            return False
 

    def _point_on_bc(self, p, tol=1.e-14):
        """
        :param p: (np.1darray) coord of a point p[0] = x, p[1] = y
        """
        x = p[0]
        y = p[1]
        # xmin BC, a vertical line
        if np.abs(x - self.xmin) < tol:
            return True
        elif np.abs(x - self.xmax) < tol:
            return True
        elif np.abs(y - self.ymin) < tol:
            return True
        elif np.abs(y - self.ymax) < tol:
            return True
        else:
            return False

    def _print(self):
        print("*****" * 4)
        print("bcMeshEdge matrix is ")
        print(self.bcMeshEdge)
        print("bcFEnode matrix is")
        print(self.bcFEnode)
        print("*****" * 4)







def test_mesh1d():
    nodes1d = np.linspace(0., 4., 5)
    mesh1d = Mesh1d(x1d_nodes=nodes1d, FE_node_num=4, print_flag=True)

def test_trimesh2d():
    x1d_mesh = np.linspace(1., 7., 3)
    y1d_mesh = np.linspace(1., 7., 3)
    mesh2d = TriMesh2d(x1d_nodes=x1d_mesh, y1d_nodes=y1d_mesh, FE_node_num=10, print_flag=True)
    bc_mesh2d = BoundaryData2d(mesh2d.P, mesh2d.CT, mesh2d.Pb, mesh2d.CTb, print_flag=True)
    bc_mesh2d.set_bctype(mesh2d.P, mesh2d.Pb, bc_name='top', min=2., max=7, bctype=1)
    bc_mesh2d._print()

if __name__ == '__main__':
    # test_mesh1d()
    test_trimesh2d()
