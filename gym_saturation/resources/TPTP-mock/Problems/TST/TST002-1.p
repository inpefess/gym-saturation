%this is a test problem for which one needs paramodulation
cnf(a1,hypothesis,
    ( q(a) )).
cnf(a2,hypothesis,
    ( ~ q(a) | f(X) = X )).
cnf(a3,hypothesis,
    ( p(X) | p(f(a)) )).
cnf(c,negated_conjecture,
    ( ~ p(X) | ~ p(f(X)) )).
