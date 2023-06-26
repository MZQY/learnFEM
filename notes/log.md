# log

* 2023.05
  * Found a package  https://people.math.sc.edu/Burkardt/c_src/fem2d_pack/fem2d_pack.html. It deals with the element and the relating Gaussian quadrature what is exactly what I need.
  * https://defelement.com/, this website provides basis functions of the triangle reference element.
  * `basis.md`
    * Add notes on Lagrange basis functions and Gaussian quadrature.
  * `basis.pyx`
    * Finished functions about 1-D basis functions with 2, 3 and 4 nodes on the interval [-1,1].
    * Finished the function about 1-D Gauss quadrature.
    * Finished functions about 2-D triangle basis functions with 3, 6 and 10 nodes.
    * Finished the function about 2-D Gauss quadrature.
  * `test_basis.ipynb`
    * Test `basis.pyx` in this notebook. 
    * Benchmarked the Gaussian quadrature of 1-D and 2-D.
* 2023.06
  * Finish `mesh.py`.
    * The `mesh.py`  supports 1-D nonuniform grid with FE node number equals 2 or 3 or 4 in a single mesh element. 
    * The `mesh.py` supports 2-D nonuniform rectangular mesh grid with FE node number equals 3 or 6 or 10 in a single triangle mesh element.
    * The `mesh.py` generates the necessary informations about mesh and FE nodes for the FEM matrix assembly, i.e. $N_m$ (total mesh node number), $N$ (total mesh element number), $N_l$ (local mesh node number), $N_{lb}$ (local FE node number),  $P$ (mesh coordinate matrix),  $T$ (mesh element node number matrix), $P_b$ (FE node coordinate matrix) ,  $T_b$  (FE node number matrix), and C-form $T$ and $T_b$ matrix (node number index starts from zero).
  * Finish `fem2d.pyx`
    * The assembly of the stiffness matrix and the load vector.
    * Found a lot of problems when using Cython with the nogil flag.
