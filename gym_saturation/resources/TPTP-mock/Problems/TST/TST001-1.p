%this is a test TPTP problem file
include('Axioms/TST001-0.ax').
cnf(test_formula,hypothesis,
    ( this_is_a_test_case(test_constant) )).
cnf(test_formula,hypothesis,
    ( ~ this_is_a_test_case(test_constant) )).
