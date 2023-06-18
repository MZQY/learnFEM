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
        self.T = np.arange(x1d_nodes.shape[0]) + 1
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
        Tb = np.zeros((self.Nlb, self.N), dtype=int)
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
        elif self.Nlb == 6:
            self.Nb = (self.nx + 1 + self.nx) * (self.ny + 1 + self.ny)
        elif self.Nlb == 10:
            self.Nb = (self.nx + 1 + self.nx * 2) * (self.ny + 1 + self.ny * 2)
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
        T = np.zeros((3, self.N), dtype=int)
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
        Tb = np.zeros((self.Nlb, self.N), dtype=int)
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



def test_mesh1d():
    nodes1d = np.linspace(0., 4., 5)
    mesh1d = Mesh1d(x1d_nodes=nodes1d, FE_node_num=4, print_flag=True)

def test_trimesh2d():
    x1d_mesh = np.linspace(1., 7., 3)
    y1d_mesh = np.linspace(1., 7., 3)
    mesh1d = TriMesh2d(x1d_nodes=x1d_mesh, y1d_nodes=y1d_mesh, FE_node_num=10, print_flag=True)

if __name__ == '__main__':
    # test_mesh1d()
    test_trimesh2d()
