# FEM data structure

The choice of the data structure for FEM directly determines the scalability and versatility of the program, highlighting the importance of selecting an appropriate form.

## symbol definitions

* $N$ 

  * (`int`) number of mesh elements
  * mesh elements are represented as $E_n,\quad n=1,2,...,N$

* $N_m$

  * (`int`) node number of mesh. Subscript 'm' for mesh
  * nodes on mesh elements are denoted as $Z_k, \quad, k=1,2,...,N_m$ 

* $N_l$

  * (`int`) node number on a single mesh element, subscript 'l' for local.

* $N_{lb}$

  * (`int`) basis function number on a single mesh element, which is also the number of FE nodes on the single mesh element. Subscript 'lb' for local basis.

* $N_b$

  * (`int`) number of FE basis functions, which is also the number of FE nodes. Subscript 'b' for basis.
  * FE nodes are denoted as $X_j, \quad j=1,2,...,N_b$

* $P$ 

  * (`matrix`) mesh coordinate matrix
    $$
    \begin{pmatrix}
    x_1  & x_2 & ... & x_k & ... & x_{N_m}\\
    y_1  & y_2 & ... & y_k & ... & y_{N_{m}}
    \end{pmatrix}
    $$

  * $P$ has n-dim rows and $N_m$ columns. The $k$-th column of $P$ is the coordinates of the $k$-th mesh node.

* T

  * (`matrix`) mesh element node number matrix

    $\begin{array}{ll}  \left(\begin{array}{cccc}    n_1 & n_2 & \ldots & n_?\\   n_4 & n_4 & \ldots & n_{? ?}\\    n_2 & n_5 & \ldots & n_{? ?}  \end{array}\right) & n \in [1, N_m]\\  \begin{array}{lllll}    & E_1 & E_2 & \ldots  & E_N  \end{array} & \end{array}$

  * $T$ matrix has $N_l$ rows and $N$ columns. The $k$-th column of $T$ is the node numbers of nodes in the $k$-th mesh element. The local node number determines the sequence of rows.

* $P_b$

  * (`matrix`) FE node coordinate matrix
    $$
    \begin{pmatrix}
    x_1  & x_2 & ... & x_k & ... & x_{N_b}\\
    y_1  & y_2 & ... & y_k & ... & y_{N_{b}}
    \end{pmatrix}
    $$

  * $P_b$ has n-dim rows and $N_b$ columns. The $k$-th column of $P_b$ is the coordinates of $k$-th global FE nodes.

  * $P_b$ determines the upper and lower limit of the FE integral.

* $T_b$ 

  * (`matrix`) FE node number matrix
  * To find the global number of the corresponding local basis functions for matrix assembly.
  * The global FE node number  of $s$-th local node on the $n$-th element: $p_s=T_b (s,n),\quad s=1,2,...,N_{lb}$

* $boundarynodes(1,k)$

  * (`matrix`) boundary node number matrix
  * 1 is the boundary type; $k$ is the global number

* $nbn$

  * number of boundary nodes

* $\varphi_{n\alpha}^{(r)}, \psi_{n\beta}^{(s)}$

  * $\varphi_{n\alpha}$ denotes the local trial basis function on $\alpha$-th local node of the $n$-th element
  * $\psi_{n\beta}$ denotes the local test basis function on $\beta$-th local node of the $n$-th element
  * In $A_{ij}$, $i$ corresponds to $i=T_b(\beta,n)$, $j$ corresponds to $j=T_b(\alpha, n)$
  * $r, s$ are the derivative orders
  * $\varphi=\psi$ (trial basis function $=$ test basis function) corresponds to Bubnov-Galerkin method; $\varphi \neq \psi$ (trial basis function $\neq$ test basis function) corresponds to Petrov-Galerkin method.



## 1D

$N$ elements has $N+1$ mesh points and $(N+1)+(p-2)\times N$ FE points, where $p$ is the internal FE point number in each element.



## 2D triangle element (uniform rectangular mesh)

If a  rectangular mesh has $N_x$ element in x-dir and $N_y$ element in y-dir, then it will have $(N_x + N_y)\times 2$ element sides on boundaries and $N_x \times N_y$ element side inside (every small rectangle is divided to two triangle elements by a inside segment). If the triangle element has $p$ inner points on every side in addition to vertices, then the rectangular mesh has $\left[ (N_x + N_y)\times 2 + N_x\times N_y \right] \times p + (N_x+1)\times(N_y+1)$  (points on inner side and vertices) FE points.

```python
"""

"""
```



