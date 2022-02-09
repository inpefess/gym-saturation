# Copyright 2021-2022 Boris Shminke

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Factoring
==========
"""
from typing import Tuple

from gym_saturation import grammar as gram
from gym_saturation.logic_ops.unification import (
    NonUnifiableError,
    most_general_unifier,
)


def factoring(
    given_clause: gram.Clause,
    literal_one: gram.Literal,
    literal_two: gram.Literal,
) -> gram.Clause:
    r"""
    .. _factoring:

    positive factoring rule

    .. math:: {\frac{C\vee A_1\vee A_2}{\sigma\left(C\vee L_1\right)}}

    where

    * :math:`C` and is a clause
    * :math:`A_1` and :math:`A_2` are atomic formulae (positive literals)
    * :math:`\sigma` is a most general unifier of :math:`A_1` and :math:`A_2`

    >>> from gym_saturation.grammar import Predicate, Variable, Function
    >>> factoring(gram.Clause((gram.Literal(True, Predicate("q", (Variable("X"),))),)), gram.Literal(False, Predicate("p", (Variable("X"),))), gram.Literal(False, Predicate("p", (Function("this_is_a_test_case", ()),)))).literals
    (Literal(negated=True, atom=Predicate(name='q', arguments=(Function(name='this_is_a_test_case', arguments=()),))), Literal(negated=False, atom=Predicate(name='p', arguments=(Function(name='this_is_a_test_case', arguments=()),))))
    >>> factoring(gram.Clause(()), gram.Literal(False, Predicate("f", ())), gram.Literal(True, Predicate("this_is_a_test_case", ())))
    Traceback (most recent call last):
     ...
    ValueError: factoring is not possible for Literal(negated=False, atom=Predicate(name='f', arguments=())) and Literal(negated=True, atom=Predicate(name='this_is_a_test_case', arguments=()))

    :param given_clause: :math:`C`
    :param literal_one: :math:`A_1`
    :param literal_two: :math:`A_2`
    :returns: a new clause --- the factoring result
    """
    if literal_one.negated or literal_two.negated:
        raise ValueError(
            f"factoring is not possible for {literal_one} and {literal_two}"
        )
    substitutions = most_general_unifier((literal_one.atom, literal_two.atom))
    new_literals = given_clause.literals + (literal_one,)
    result = gram.Clause(new_literals)
    for substitution in substitutions:
        result = substitution.substitute_in_clause(result)
    return result


def all_possible_factors(
    given_clause: gram.Clause,
) -> Tuple[gram.Clause, ...]:
    """
    one of the four basic building blocks of the Given Clause algorithm

    >>> from gym_saturation.parsing.tptp_parser import TPTPParser
    >>> parser = TPTPParser()
    >>> clause = parser.parse("cnf(one, axiom, p(c) | p(X) | q).", "")[0]
    >>> all_possible_factors(clause)  # doctest: +ELLIPSIS
    (cnf(x..., lemma, q() | p(c), inference(factoring, [], [one])).,)

    :param given_clause: a new clause which should be combined with all the
        processed ones
    :returns: results of all possible factors with each one from
        ``clauses`` and the ``given_clause``
    """
    factors: Tuple[gram.Clause, ...] = ()
    for i, literal_one in enumerate(given_clause.literals):
        for j in range(i + 1, len(given_clause.literals)):
            if (
                not literal_one.negated
                and not given_clause.literals[j].negated
            ):
                a_clause = gram.Clause(
                    given_clause.literals[:i]
                    + given_clause.literals[i + 1 : j]
                    + given_clause.literals[j + 1 :]
                )
                try:
                    factors = factors + (
                        factoring(
                            a_clause, literal_one, given_clause.literals[j]
                        ),
                    )
                except NonUnifiableError:
                    pass
    return tuple(
        gram.Clause(
            literals=factor.literals,
            inference_parents=(given_clause.label,)
            if given_clause.label is not None
            else None,
            inference_rule="factoring",
        )
        for ord_num, factor in enumerate(factors)
    )
