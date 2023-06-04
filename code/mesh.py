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

def test_mesh1d():
    nodes1d = np.linspace(0., 4., 5)
    mesh1d = Mesh1d(x1d_nodes=nodes1d, FE_node_num=4, print_flag=True)


if __name__ == '__main__':
    test_mesh1d()
