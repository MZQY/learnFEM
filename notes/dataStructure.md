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



### mesh grid/$N_{lb}=3$



![2dtriMesh](figs/2dtriMesh.svg)



element plot

![2dtriMesh_element](figs/2dtriMesh_element.svg)



```python
test case:
    x1d_mesh = np.array([1, 2, 3])
    y1d_mesh = np.array([1, 2, 3])
    
    
my code prints:
T matrix is
[[1 2 2 3 4 5 5 6]
 [4 4 5 5 7 7 8 8]
 [2 5 3 6 5 8 6 9]]
Tb matrix is
[[1 2 2 3 4 5 5 6]
 [4 4 5 5 7 7 8 8]
 [2 5 3 6 5 8 6 9]]
P matrix is
[[1. 1. 1. 2. 2. 2. 3. 3. 3.]
 [1. 2. 3. 1. 2. 3. 1. 2. 3.]]
Pb matrix is
[[1. 1. 1. 2. 2. 2. 3. 3. 3.]
 [1. 2. 3. 1. 2. 3. 1. 2. 3.]]
```





### $N_{lb}=6$

```python
    """
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
    """
```



![2dtriFE_nlb6](figs/2dtriFE_nlb6.svg)



```python
test case:
    x1d_mesh = np.array([1,  3,  5])
    y1d_mesh = np.array([1,  3,  5])
    
    
my code prints:
T matrix is
[[1 2 2 3 4 5 5 6]
 [4 4 5 5 7 7 8 8]
 [2 5 3 6 5 8 6 9]]
Tb matrix is
[[ 1  3  3  5 11 13 13 15]
 [11 11 13 13 21 21 23 23]
 [ 3 13  5 15 13 23 15 25]
 [ 7 12  9 14 17 22 19 24]
 [ 2  8  4 10 12 18 14 20]
 [ 6  7  8  9 16 17 18 19]]
P matrix is
[[1. 1. 1. 3. 3. 3. 5. 5. 5.]
 [1. 3. 5. 1. 3. 5. 1. 3. 5.]]
Pb matrix is
[[1. 1. 1. 1. 1. 2. 2. 2. 2. 2. 3. 3. 3. 3. 3. 4. 4. 4. 4. 4. 5. 5. 5. 5.
  5.]
 [1. 2. 3. 4. 5. 1. 2. 3. 4. 5. 1. 2. 3. 4. 5. 1. 2. 3. 4. 5. 1. 2. 3. 4.
  5.]]
```







### $N_{lb}=10$



```python
    """
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
    """
```



![2dtriFE_nlb10](figs/2dtriFE_nlb10.svg)



```python
test case:
    x1d_mesh = np.array([1,  7,  5])
    y1d_mesh = np.array([1,  7,  5])
    
my code prints:
    T matrix is
[[1 2 2 3 4 5 5 6]
 [4 4 5 5 7 7 8 8]
 [2 5 3 6 5 8 6 9]]
Tb matrix is
[[ 1  4  4  7 22 25 25 28]
 [22 22 25 25 43 43 46 46]
 [ 4 25  7 28 25 46 28 49]
 [16 23 19 26 37 44 40 47]
 [10 24 13 27 31 45 34 48]
 [ 2 11  5 14 23 32 26 35]
 [ 3 18  6 21 24 39 27 42]
 [ 8 10 11 13 29 31 32 34]
 [15 16 18 19 36 37 39 40]
 [ 9 17 12 20 30 38 33 41]]
P matrix is
[[1. 1. 1. 4. 4. 4. 7. 7. 7.]
 [1. 4. 7. 1. 4. 7. 1. 4. 7.]]
Pb matrix is
[[1. 1. 1. 1. 1. 1. 1. 2. 2. 2. 2. 2. 2. 2. 3. 3. 3. 3. 3. 3. 3. 4. 4. 4.
  4. 4. 4. 4. 5. 5. 5. 5. 5. 5. 5. 6. 6. 6. 6. 6. 6. 6. 7. 7. 7. 7. 7. 7.
  7.]
 [1. 2. 3. 4. 5. 6. 7. 1. 2. 3. 4. 5. 6. 7. 1. 2. 3. 4. 5. 6. 7. 1. 2. 3.
  4. 5. 6. 7. 1. 2. 3. 4. 5. 6. 7. 1. 2. 3. 4. 5. 6. 7. 1. 2. 3. 4. 5. 6.
  7.]]
```







