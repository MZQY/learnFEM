# log

* 2023.05
  * Found a package  https://people.math.sc.edu/Burkardt/c_src/fem2d_pack/fem2d_pack.html. It deals with the element and the relating Gaussian quadrature what is exactly what I need.
  * https://defelement.com/, this website provides basis functions of the triangle reference element.
  * `basis.md`
    * Add notes on Lagrange basis functions and Gaussian quadrature.
  * `basis.pyx`
    * Finished functions about 1D basis functions with 2, 3 and 4 nodes on the interval [-1,1].
    * Finished the function about 1D Gauss quadrature.
    * Finished functions about 2D triangle basis functions with 3, 6 and 10 nodes.
    * Finished the function about 2D Gauss quadrature.
  * `test_basis.ipynb`
    * Test `basis.pyx` in this notebook. 
    * Benchmarked the Gaussian quadrature of 1D and 2D.
