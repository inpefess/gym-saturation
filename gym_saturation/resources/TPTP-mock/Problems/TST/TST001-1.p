%this is a test TPTP problem file
include('Axioms/TST001-0.ax').
cnf(this_is_a_test_case_1,hypothesis,
    ( this_is_a_test_case(test_constant) ), inference(resolution,[],[one,two])).
cnf(this_is_a_test_case_2,hypothesis,
    ( ~ this_is_a_test_case(test_constant) )).
%----Comments
/* This
   is a block
   comment.
*/
