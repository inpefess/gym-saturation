cnf(associativity, axiom, mult(X, mult(Y, Z)) = mult(mult(X, Y), Z)).
cnf(left_identity, axiom, mult(e, X) = X).
cnf(left_inverse, axiom, mult(inv(X), X) = e).
cnf(idempotent_element, hypothesis, mult(a, a) = a).
cnf(negated_conjecture, negated_conjecture, ~ a = e).
