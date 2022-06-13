%this is a test TPTP problem file
include('Axioms/TST001-0.ax').
cnf(this_is_a_test_case_1,hypothesis,
    ( p(c) ), inference(resolution,[],[one,two])).
cnf(this_is_a_test_case_2,hypothesis,
    ( ~ p(c) )).
%----Comments
/* This
   is a block
   comment.
*/
