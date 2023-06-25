<TeXmacs|2.1>

<style|article>

<\body>
  <doc-data|<doc-title|2D 2-order elliptic
  equations>|<doc-author|<author-data|<author-name|ymma98>|<\author-affiliation>
    <date|>
  </author-affiliation>>>>

  <\table-of-contents|toc>
    <vspace*|1fn><with|font-series|bold|math-font-series|bold|1<space|2spc>Hilbert
    spaces> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-1><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|2<space|2spc>Target
    problems> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-2><vspace|0.5fn>

    <with|par-left|1tab|2.1<space|2spc>Dirichilet boundary condition
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-3>>

    <with|par-left|1tab|2.2<space|2spc> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-4>>
  </table-of-contents>

  <section|Hilbert spaces>

  From a mathematical point of view, the \Pright\Q choice of function space
  is essential since this may make it easier to <with|color|red|<strong|prove
  the existence of a solution<strong|>>> to the continuous problem.

  A weak formulation is obtained by multiplying the original equation by test
  functions and then integrate it. <with|color|red|The advantage of the weak
  formulation is that it is easy to prove the existence of a solution>.

  <\definition>
    Linear form.

    If <math|V> is a linear space, we say that <math|L> is a linear form in
    <math|V> if <math|L:<space|0.6spc>V\<rightarrow\>\<bbb-R\>>, i.e.,
    <math|L<around*|(|v|)>\<in\>\<bbb-R\>> for <math|v\<in\>V>, and <math|L>
    is linear, i.e. for all <math|v,w\<in\>V> and
    <math|\<beta\>,\<theta\>\<in\>\<bbb-R\>>,

    <\equation*>
      L<around*|(|\<beta\>v+\<theta\>w|)>=\<beta\>L<around*|(|v|)>+\<theta\>L<around*|(|w|)>
    </equation*>
  </definition>

  <\definition>
    Bilinear form.

    <math|a<around*|(|\<cdot\>,\<cdot\>|)>> is a bilinear form on
    <math|V\<times\>V> if <math|a:<space|0.6spc>V\<times\>V\<rightarrow\>\<bbb-R\>>.
    i.e., <math|a<around*|(|v,w|)>\<in\>\<bbb-R\>> for <math|v,w\<in\>V>, and
    <math|a<around*|(|\<cdot\>,\<cdot\>|)>> is linear in each argument, i.e.
    for all <math|u,v,w\<in\>V> and <math|><math|\<beta\>,\<theta\>\<in\>\<bbb-R\>>,
    we have

    <\equation*>
      <tabular|<tformat|<table|<row|<cell|a<around*|(|u,\<beta\>v+\<theta\>w|)>>|<cell|=>|<cell|\<beta\>a<around*|(|u,v|)>+\<theta\>a<around*|(|u,w|)>>>|<row|<cell|a<around*|(|\<beta\>u+\<theta\>v,w|)>>|<cell|=>|<cell|\<beta\>a<around*|(|u,w|)>+\<theta\>a<around*|(|v,w|)>>>>>>
    </equation*>

    The bilinear form <math|a<around*|(|\<cdot\>,\<cdot\>|)>> is symmetric on
    <math|V\<times\>V> if <math|a<around*|(|v,w|)>=a<around*|(|w,v|)>>.
  </definition>

  <\definition>
    Scalar product.

    A scalar product is a symmetric bilinear form
    <math|a<around*|(|\<cdot\>,\<cdot\>|)>> on <math|V\<times\>V> where
    <math|a<around*|(|\<cdot\>,\<cdot\>|)>\<gtr\>0<space|1em>\<forall\>v\<in\>V,v\<neq\>0>
  </definition>

  <\definition>
    Norm.

    The norm <math|<around*|\<\|\|\>|\<cdot\>|\<\|\|\>><rsub|a>> associated
    with a scalar product <math|a<around*|(|\<cdot\>,\<cdot\>|)>> is defined
    by\ 

    <\equation*>
      <around*|\<\|\|\>|v|\<\|\|\>><rsub|a>=<around*|[|a<around*|(|v,v|)>|]><rsup|1/2><space|1em>\<forall\>v\<in\>V
    </equation*>

    (<with|color|red|bilinear product <math|\<rightarrow\>> scalar product
    \<rightarrow\> norm>)
  </definition>

  <\theorem>
    Cauchy's inequality.

    If <math|\<less\>\<cdot\>,\<cdot\>\<gtr\>> is a scalar product with
    corresponding norm <math|<around*|\<\|\|\>|\<cdot\>|\<\|\|\>><rsub|>>,
    then we have Cauchy's inequality

    <\equation*>
      <around*|\||\<less\>v,w\<gtr\>|\|>\<leqslant\><around*|\<\|\|\>|v|\<\|\|\>>
      <around*|\<\|\|\>|w|\<\|\|\>>
    </equation*>
  </theorem>

  <\definition>
    Hilbert space.

    If <math|V> is a linear space with a scalar product with corresponding
    norm <math|<around*|\<\|\|\>|\<cdot\>|\<\|\|\>>>, then <math|V> is said
    to be a Hilbert space if <math|V> is complete, i.e., if every Cauchy
    sequence with respect to <math|<around*|\<\|\|\>|\<cdot\>|\<\|\|\>>> is
    convergent.

    We could just think a Hilbert space simply as a linear space with a
    scalar product (a special positive symmetric bilinear form).
  </definition>

  <\remark>
    Cauchy sequence.

    A sequence <math|v<rsub|1>,v<rsub|2>,v<rsub|3>,\<ldots\>> of elements
    <math|v<rsub|i>> in the space <math|V> with norm
    <math|<around*|\<\|\|\>|\<cdot\>|\<\|\|\>>> is said to be a Cauchy
    sequence if for all <math|\<epsilon\>\<gtr\>0>, there is a natrual number
    <math|N> such that <math|<around*|\<\|\|\>|v<rsub|i>-v<rsub|j>|\<\|\|\>>\<less\>\<epsilon\>>
    if <math|i,j\<gtr\>N>. Further, <math|v<rsub|i>> converges to <math|v> if
    <math|<around*|\<\|\|\>|v-v<rsub|i>|\<\|\|\>>\<rightarrow\>0> as
    <math|i\<rightarrow\>\<infty\>>.
  </remark>

  <\definition>
    <math|C<rsub|0><rsup|\<infty\>><around*|(|\<Omega\>|)>> is the set of all
    functions that are infinitely differentiable on <math|\<Omega\>> and
    compactly supported in <math|\<Omega\>>.
  </definition>

  <\definition>
    <math|L<rsup|p> space> (finite energy space)

    <\equation*>
      L<rsup|p><around|(|\<Omega\>|)>=<around*|{|v:\<Omega\>\<rightarrow\><math-bf|R>:<big|int><rsub|\<Omega\>><around|\||v|\|><rsup|p>*d*x*d*y\<less\>\<infty\>|}>
    </equation*>
  </definition>

  <\corollary>
    <math|L<rsup|2>> space

    <\equation*>
      L<rsup|2><around|(|\<Omega\>|)>=<around*|{|v:\<Omega\>\<rightarrow\><math-bf|R>:<big|int><rsub|<around*|(|x,y|)>\<in\>\<Omega\>>v<rsup|2>*d*x*d*y\<less\>\<infty\>|}>
    </equation*>

    The <math|L<rsup|2><around|(|\<Omega\>|)>> space is a Hilbert space with
    the scalar product

    <\equation*>
      <around*|(|v,w|)>=<big|int><rsub|x\<in\>\<Omega\>>v w d x
    </equation*>

    and the corresponding norm (the <math|L<rsup|2>>-norm)

    <\equation*>
      <around*|\<\|\|\>|v|\<\|\|\>><rsub|L<rsup|2><around*|(|\<Omega\>|)>>=<around*|[|<big|int><rsub|x\<in\>\<Omega\>>v<rsup|2>d
      x|]><rsup|1/2>=<around*|(|v,v|)><rsup|1/2>
    </equation*>
  </corollary>

  <\definition>
    <math|L<rsup|\<infty\>>> space

    <\equation*>
      L<rsup|\<infty\>><around|(|\<Omega\>|)>=<around*|{|v:\<Omega\>\<rightarrow\><math-bf|R>:<big|int><rsub|\<Omega\>>sup
      <around*|\||v|\|>\<less\>\<infty\>*d*x*d*y\<less\>\<infty\>|}>
    </equation*>
  </definition>

  <\definition>
    <math|H<rsup|m>> space

    <\equation*>
      H<rsup|m><around|(|\<Omega\>|)>=<around*|{|v\<in\>L<rsup|2><around|(|\<Omega\>|)>:<frac|\<partial\><rsup|\<alpha\><rsub|1>+\<alpha\><rsub|2>>*v|\<partial\>*x<rsup|\<alpha\><rsub|1>>*\<partial\>*y<rsup|\<alpha\><rsub|2>>>\<in\>L<rsup|2><around|(|\<Omega\>|)>,\<forall\>\<alpha\><rsub|1>+\<alpha\><rsub|2>=1,\<cdots\>,m|}>
    </equation*>
  </definition>

  <\corollary>
    <math|H<rsup|1>> space (<math|v> and <math|v<rsup|<rprime|'>>> belong to
    <math|L<rsup|2><around*|(|\<Omega\>|)>>)

    <\equation*>
      H<rsup|1><around|(|\<Omega\>|)>=<around*|{|v\<in\>L<rsup|2><around|(|\<Omega\>|)>:<frac|\<partial\><rsup|\<alpha\><rsub|1>+\<alpha\><rsub|2>>*v|\<partial\>*x<rsup|\<alpha\><rsub|1>>*\<partial\>*y<rsup|\<alpha\><rsub|2>>>\<in\>L<rsup|2><around|(|\<Omega\>|)>,\<forall\>\<alpha\><rsub|1>+\<alpha\><rsub|2>=1|}>
    </equation*>

    The <math|H<rsup|1>> space has the scalar product

    <\equation*>
      <around*|(|v,w|)><rsub|H<rsup|1><around*|(|\<Omega\>|)>>=<big|int><rsub|x\<in\>\<Omega\>><around*|(|v
      w+v<rsup|<rprime|'>>w<rsup|<rprime|'>>|)> d x
    </equation*>

    and the corresponding norm

    <\equation*>
      <around*|\<\|\|\>|v|\<\|\|\>><rsub|H<rsup|1><around*|(|\<Omega\>|)>>=<around*|[|<big|int><rsub|x\<in\>\<Omega\>><around*|(|v<rsup|2>+<around*|(|v<rsup|<rprime|'>>|)><rsup|2>|)>d
      \<Omega\>|]><rsup|1/2>
    </equation*>
  </corollary>

  <\definition>
    <math|H<rsub|0><rsup|1>> space

    <\equation*>
      H<rsub|0><rsup|1><around|(|\<Omega\>|)>=<around*|{|v\<in\>H<rsup|1><around|(|\<Omega\>|)>:v=0
      on \<partial\>\<Omega\>|}>
    </equation*>
  </definition>

  <\definition>
    <math|W<rsub|p><rsup|m>> space

    <\equation*>
      <\aligned>
        <tformat|<table|<row|<cell|W<rsub|p><rsup|m><around|(|\<Omega\>|)>=>|<cell|<around*|{|v:\<Omega\>\<rightarrow\><math-bf|R>:<big|int><rsub|\<Omega\>><around*|[|<frac|\<partial\><rsup|\<alpha\><rsub|1>+\<alpha\><rsub|2>>*v|\<partial\>*x<rsup|\<alpha\><rsub|1>>*\<partial\>*y<rsup|\<alpha\><rsub|2>>>|]><rsup|p>*d*x*d*y\<less\>\<infty\>,|\<nobracket\>>>>|<row|<cell|>|<cell|<around*|\<nobracket\>|\<forall\>\<alpha\><rsub|1>+\<alpha\><rsub|2>=0,\<cdots\>,m|}>.>>>>
      </aligned>
    </equation*>
  </definition>

  <\remark>
    \;

    <\equation*>
      <\aligned>
        <tformat|<table|<row|<cell|>|<cell|L<rsup|p><around|(|\<Omega\>|)>=W<rsub|p><rsup|0><around|(|\<Omega\>|)>;>>|<row|<cell|>|<cell|L<rsup|2><around|(|\<Omega\>|)>=W<rsub|2><rsup|0><around|(|\<Omega\>|)>;>>|<row|<cell|>|<cell|H<rsup|m><around|(|\<Omega\>|)>=W<rsub|2><rsup|m><around|(|\<Omega\>|)>;>>|<row|<cell|>|<cell|H<rsup|1><around|(|\<Omega\>|)>=W<rsub|2><rsup|1><around|(|\<Omega\>|)>>>>>
      </aligned>
    </equation*>
  </remark>

  <section|Target problem>

  Consider the 2D 2-order elliptic equation

  <\equation>
    -\<nabla\>\<cdot\><around*|(|c\<nabla\>u|)>=f<space|1em>in \<Omega\>
  </equation>

  Firstly, multiply the test function <math|v> on both sides of the original
  equation and then take the integration,

  <\equation>
    <tabular|<tformat|<cwith|1|-1|1|1|cell-halign|r>|<table|<row|<cell|-\<nabla\>\<cdot\><around*|(|c\<nabla\>u|)>v>|<cell|=>|<cell|f
    v>>|<row|<cell|-<big|int><rsub|\<Omega\>>\<nabla\>\<cdot\><around*|(|c\<nabla\>u|)>v
    d \<Omega\>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>f v d\<Omega\>>>>>>
  </equation>

  Secondly, using the integration by parts,

  <\equation>
    <tabular|<tformat|<cwith|1|-1|1|1|cell-halign|r>|<table|<row|<cell|-<big|int><rsub|\<Omega\>>\<nabla\>\<cdot\><around*|(|c\<nabla\>u|)>v
    d \<Omega\>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>f v
    d\<Omega\>>>|<row|<cell|-<around*|(|<big|int><rsub|\<Omega\>>\<nabla\>\<cdot\><around*|(|v
    c\<nabla\>u|)> d \<Omega\>-<big|int><rsub|\<Omega\>>c\<nabla\>u\<cdot\>\<nabla\>v
    d \<Omega\>|)>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>f v
    d\<Omega\>>>|<row|<cell|<big|int><rsub|\<Omega\>>c\<nabla\>u\<cdot\>\<nabla\>v
    d \<Omega\><with|color|dark green|-<big|int><rsub|\<partial\>\<Omega\>>v
    c\<nabla\>u\<cdot\><wide|n|\<vect\>> d
    S>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>f v d\<Omega\>>>>>>
  </equation>

  Thridly, discretize the equation. The trial function has the form
  <math|u<rsub|h>=<big|sum><rsub|j=1><rsup|N<rsub|b>>u<rsub|j>\<varphi\><rsub|j>>,
  and the test function has the form <math|v<rsub|<rsub|h>>=\<psi\><rsub|i>>,

  <\equation>
    <with|color|black|<big|sum><rsub|j=1><rsup|N<rsub|b>>u<rsub|j><with|color|red|<big|int><rsub|\<Omega\>>c\<nabla\>\<varphi\><rsub|j>\<cdot\>\<nabla\>\<psi\><rsub|i>
    d \<Omega\>>><with|color|dark green|-><with|color|dark
    green|<with|color|black|<big|sum><rsub|j=1><rsup|N<rsub|b>>u<rsub|j>><big|int><rsub|\<partial\>\<Omega\>>\<psi\><rsub|i>
    c\<nabla\>\<varphi\><rsub|j>\<cdot\><wide|n|\<vect\>> d
    S>=<with|color|blue|<big|int>f\<psi\><rsub|i>d\<Omega\>>
  </equation>

  The second term is determined by the <with|color|dark cyan|<with|color|dark
  green|boundary conditions>>.

  The stiffness matrix <math|<wide|<wide|A|\<vect\>>|\<vect\>>>, has the form

  <\equation>
    <with|color|red|<wide|<wide|A|\<vect\>>|\<vect\>>=<around*|[|a<rsub|ij>|]><rsub|i,j=1><rsup|N<rsub|b>>>,<space|1em>a<rsub|ij>=<big|int><rsub|\<Omega\>>c\<nabla\>\<varphi\><rsub|j>\<cdot\>\<nabla\>\<psi\><rsub|i>
    d \<Omega\>
  </equation>

  The load vector <math|<wide|b|\<vect\>>>, has the form

  <\equation>
    <with|color|blue|<wide|b|\<vect\>>=<around*|[|b<rsub|i>|]><rsub|i=1><rsup|N<rsub|b>>>,<space|1em>b<rsub|i>=<big|int>f\<psi\><rsub|i>d\<Omega\>
  </equation>

  The unknown vector <math|<wide|x|\<vect\>>>, has the form

  <\equation>
    <wide|x|\<vect\>>=<around*|[|u<rsub|j>|]><rsub|j=1><rsup|N<rsub|b>>
  </equation>

  Thus the differential equation converts to the matrix form
  <math|<wide|<wide|A|\<vect\>>|\<vect\>>
  <wide|x|\<vect\>>=<wide|b|\<vect\>>>.

  <subsection|Dirichilet boundary condition>

  The Dirichlet BC has the form

  <\equation>
    u=g<space|1em>on \<partial\>\<Omega\>
  </equation>

  Since the values of <math|u> on the boundary are known, whatever form of
  the <math|<big|int><rsub|\<partial\>\<Omega\>>\<psi\><rsub|i>
  c\<nabla\>\<varphi\><rsub|j>\<cdot\><wide|n|\<vect\>> d S> term will not
  affect the final results (we always set the <math|u<rsub|j>> corresponding
  to the Dirichlet BC to the exact value). It is sensible to set the term
  <math|<big|int><rsub|\<partial\>\<Omega\>>\<psi\><rsub|i>
  c\<nabla\>\<varphi\><rsub|j>\<cdot\><wide|n|\<vect\>> d S> to zero.

  <\equation>
    \<psi\><rsub|i>=0<space|1em>on \<partial\>\<Omega\>
  </equation>

  Thus the equation turns out to be

  <\equation>
    <tabular|<tformat|<cwith|1|1|1|1|cell-halign|r>|<table|<row|<cell|<big|int><rsub|\<Omega\>>c\<nabla\>u\<cdot\>\<nabla\>v
    d \<Omega\>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>f v d\<Omega\>>>>>>
  </equation>

  The pseudo code has the form:

  <\render-code>
    for <math|k=1,\<ldots\>.,nbn>

    <space|3em>if boundarynodes(1, <math|k>) shows Dirichlet condition, then

    <space|5em><math|i=boundarynodes<around*|(|2,k|)>>

    <space|5em><math|A>(<math|i>,:)=0

    <space|5em><math|A>(<math|i>,<math|i>)=1

    <space|5em><math|b<around*|(|i|)>=g<around*|(|P<rsub|b><around*|(|:,i|)>|)>>

    <space|3em>endif

    endfor
  </render-code>

  <subsection|Neumann boundary condition>

  The Neumann BC has the form

  <\equation>
    \<nabla\>u\<cdot\><wide|n|\<vect\>>=p
  </equation>

  Thus the boundary condition term <math|<big|int><rsub|\<partial\>\<Omega\>>v
  c\<nabla\>u\<cdot\><wide|n|\<vect\>> d S>

  <\equation>
    <big|int><rsub|\<partial\>\<Omega\>>v
    c\<nabla\>u\<cdot\><wide|n|\<vect\>> d
    S=<big|int><rsub|\<partial\>\<Omega\>> c p v d S
  </equation>

  And the target equation turns out to be

  <\equation>
    <tabular|<tformat|<cwith|1|1|1|1|cell-halign|r>|<cwith|2|2|1|1|cell-hyphen|n>|<cwith|2|2|1|1|cell-halign|r>|<cwith|3|3|1|1|cell-hyphen|t>|<cwith|3|3|1|1|cell-halign|r>|<table|<row|<cell|<big|int><rsub|\<Omega\>>c\<nabla\>u\<cdot\>\<nabla\>v
    d \<Omega\><with|color|dark green|-<big|int><rsub|\<partial\>\<Omega\>>v
    c\<nabla\>u\<cdot\><wide|n|\<vect\>> d
    S>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>f v
    d\<Omega\>>>|<row|<cell|<big|int><rsub|\<Omega\>>c\<nabla\>u\<cdot\>\<nabla\>v
    d \<Omega\>-<with|color|dark green|<big|int><rsub|\<partial\>\<Omega\>> c
    p v d S>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>f v
    d\<Omega\>>>|<row|<\cell>
      <big|int><rsub|\<Omega\>>c\<nabla\>u\<cdot\>\<nabla\>v d \<Omega\>
    </cell>|=|<cell|<big|int><rsub|\<Omega\>>f v d\<Omega\>+<with|color|dark
    green|<big|int><rsub|\<partial\>\<Omega\>> c p v d S>>>>>>
  </equation>

  After the discretization\ 

  <\equation>
    <big|int><rsub|\<partial\>\<Omega\>> c p v d
    S=<big|int><rsub|\<partial\>\<Omega\>> c p \<psi\><rsub|i> d S
  </equation>

  Thus the BC term adds to the load vector.

  <\equation>
    b<rsub|i,final>=b<rsub|i>+<big|int><rsub|\<partial\>\<Omega\>> c p
    \<psi\><rsub|i> d S
  </equation>

  <subsection|Robin boundary condition>

  The Robin BC has the form

  <\equation>
    \<nabla\>u\<cdot\><wide|n|\<vect\>>+r
    u=q<space|1em>on<space|0.6spc>\<Gamma\><rsub|R>\<subset\>\<partial\>\<Omega\>
  </equation>

  Thus

  <\equation>
    <big|int><rsub|\<partial\>\<Omega\>>v
    c\<nabla\>u\<cdot\><wide|n|\<vect\>> d
    S=<big|int><rsub|\<partial\>\<Omega\>>v c<around*|(|q-r u|)> d
    S=<big|int><rsub|\<partial\>\<Omega\>>v c q d
    S-<big|int><rsub|\<partial\>\<Omega\>>v c r u d S
  </equation>

  And the target equation turns out to be

  <\equation>
    <tabular|<tformat|<cwith|1|1|1|1|cell-halign|r>|<cwith|2|2|1|1|cell-hyphen|n>|<cwith|2|2|1|1|cell-halign|r>|<cwith|3|3|1|1|cell-hyphen|t>|<cwith|3|3|1|1|cell-halign|r>|<table|<row|<cell|<big|int><rsub|\<Omega\>>c\<nabla\>u\<cdot\>\<nabla\>v
    d \<Omega\><with|color|dark green|-<big|int><rsub|\<partial\>\<Omega\>>v
    c\<nabla\>u\<cdot\><wide|n|\<vect\>> d
    S>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>f v
    d\<Omega\>>>|<row|<cell|<big|int><rsub|\<Omega\>>c\<nabla\>u\<cdot\>\<nabla\>v
    d \<Omega\>-<with|color|dark green|<around*|(|<big|int><rsub|\<partial\>\<Omega\>>v
    c q d S-<big|int><rsub|\<partial\>\<Omega\>>v c r u d
    S|)>>>|<cell|=>|<cell|<big|int><rsub|\<Omega\>>f v
    d\<Omega\>>>|<row|<\cell>
      <big|int><rsub|\<Omega\>>c\<nabla\>u\<cdot\>\<nabla\>v d
      \<Omega\>+<with|color|dark green|<big|int><rsub|\<partial\>\<Omega\>>v
      c r u d S>
    </cell>|=|<cell|<big|int><rsub|\<Omega\>>f v d\<Omega\>+<with|color|dark
    green|<big|int><rsub|\<partial\>\<Omega\>>v c q d S>>>>>>
  </equation>

  Thus after the discretization, the stiff matrix has the form

  <\equation>
    a<rsub|ij,final>=a<rsub|ij>+<with|color|dark cyan|<with|color|dark
    green|<big|int><rsub|\<partial\>\<Omega\>>c r \<psi\><rsub|i>
    \ \<varphi\><rsub|j> d S>>
  </equation>

  and the load vector has the form

  <\equation>
    b<rsub|i,final>=b<rsub|i>+<with|color|dark yellow|<with|color|dark
    green|<big|int><rsub|\<partial\>\<Omega\>>\<psi\><rsub|i> c q d S>>
  </equation>

  <section|FEM discretization>

  For the stiffness matrix in the Cartesian coordinate, the matrix element
  has the form

  <\equation>
    a<rsub|ij>=<big|int><rsub|\<Omega\>>c\<nabla\>\<varphi\><rsub|j>\<cdot\>\<nabla\>\<psi\><rsub|i>
    d \<Omega\>=<big|int><rsub|\<Omega\>>c<frac|\<partial\>\<varphi\><rsub|j>|\<partial\>x><frac|\<partial\>\<psi\><rsub|i>|\<partial\>x>
    d \<Omega\>+<big|int><rsub|\<Omega\>>c<frac|\<partial\>\<varphi\><rsub|j>|\<partial\>y><frac|\<partial\>\<psi\><rsub|i>|\<partial\>y>
    d \<Omega\>=A<rsub|1>+A<rsub|2>
  </equation>

  For the load vector in the Cartesian coordinate, the vector element has the
  form

  <\equation>
    b<rsub|i>=<big|int>f\<psi\><rsub|i>d\<Omega\>
  </equation>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|3>>
    <associate|auto-3|<tuple|2.1|3>>
    <associate|auto-4|<tuple|2.2|4>>
    <associate|auto-5|<tuple|2.3|4>>
    <associate|auto-6|<tuple|3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Hilbert
      spaces> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Target
      problem> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Dirichilet boundary
      condition <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Neumann boundary condition
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|2.3<space|2spc>Robin boundary condition
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>
    </associate>
  </collection>
</auxiliary>