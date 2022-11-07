%this is a test TPTP problem file
include('Axioms/TST001-0.ax').
cnf(hypothesys1,hypothesis,
    ( p(c) ), inference(resolution,[],[one,two])).
cnf(hypohesys2,hypothesis,
    ( ~ p(c) )).
%----Comments
/* This
   is a block
   comment.
*/
