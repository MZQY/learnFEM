function sample_v = local_fem_1d ( order, node_x, node_v, sample_num, sample_x )

%*****************************************************************************80
%
%% local_fem_1d() evaluates a local finite element function.
%
%  Discussion:
%
%    A local finite element function is a finite element function
%    defined over a single element.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    01 March 2011
%
%  Author:
%
%    John Burkardt
%
%  Input:
%
%    integer ORDER, the order of the element.
%    0 <= ORDER.  ORDER = 1 means piecewise linear.
%
%    real NODE_X(ORDER), the element nodes.  These must be distinct.
%    Basis function I is 1 when X = NODE_X(I) and 0 when X is equal
%    to any other node.
%
%    real NODE_V(ORDER), the value of the finite element function
%    at each node.
%
%    integer SAMPLE_NUM, the number of sample points.
%
%    real SAMPLE_X(SAMPLE_NUM), the sample points at which the
%    local finite element function is to be evaluated.
%
%  Output:
%
%    real SAMPLE_V(SAMPLE_NUM), the values of the local finite element
%    basis functions.
%
  sample_v = zeros ( sample_num, 1 );

  for sample = 1 : sample_num

    x = sample_x(sample);
    phi = local_basis_1d ( order, node_x, x );
    sample_v(sample) = node_v(1:order)' * phi(1:order);

  end

  return
end
