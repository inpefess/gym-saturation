# Copyright 2021-2022 Boris Shminke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# noqa: D205, D400
"""
Resolution
===========
"""
from typing import Tuple

from tptp_lark_parser.grammar import Clause, Literal

from gym_saturation.logic_ops.unification import (
    NonUnifiableError,
    most_general_unifier,
)


def resolution(
    clause_one: Clause,
    literal_one: Literal,
    clause_two: Clause,
    literal_two: Literal,
) -> Clause:
    r"""
    Do inference according to the standard first-order resolution rule.

    .. _resolution:

    .. math:: {\frac{C_1\vee L_1,C_2\vee L_2}{\sigma\left(C_1\vee C_2\right)}}

    where

    * :math:`C_1` and :math:`C_2` are clauses with no common variables
    * :math:`L_1` and :math:`L_2` are literals (one negated and one not)
    * :math:`\sigma` is a most general unifier of atoms from :math:`L_1` and :math:`L_2`

    >>> from tptp_lark_parser.grammar import Predicate, Variable, Function
    >>> "this_is_a_test_case", resolution(
    ...     Clause((Literal(False, Predicate(4, (Variable(0),))),)),
    ...     Literal(False, Predicate(3, (Variable(0),))),
    ...     Clause((Literal(False, Predicate(5, (Variable(0),))),)),
    ...     Literal(True, Predicate(3, (Function(1, ()),)))
    ... ).literals
    ('this_is_a_test_case', (Literal(negated=False, atom=Predicate(index=4, arguments=(Function(index=1, arguments=()),))), Literal(negated=False, atom=Predicate(index=5, arguments=(Function(index=1, arguments=()),)))))
    >>> resolution(
    ...     Clause(()), Literal(False, Predicate(8, ())),
    ...     Clause(()), Literal(False, Predicate(7, ()))
    ... )
    Traceback (most recent call last):
     ...
    ValueError: resolution is not possible for Literal(negated=False, atom=...

    :param clause_one: :math:`C_1`
    :param literal_one: :math:`L_1`
    :param clause_two: :math:`C_2`
    :param literal_two: :math:`L_2`
    :returns: a new clause --- the resolution result
    :raises ValueError: if both ``literal_one`` and ``literal_two`` are negated
        or non-negated
    """
    if literal_one.negated == literal_two.negated:
        raise ValueError(
            f"resolution is not possible for {literal_one} and {literal_two}"
        )
    substitutions = most_general_unifier((literal_one.atom, literal_two.atom))
    new_literals = clause_one.literals
    for literal in clause_two.literals:
        if literal not in new_literals:
            new_literals = new_literals + (literal,)
    result = Clause(new_literals)
    for substitution in substitutions:
        result = substitution.substitute_in_clause(result)
    return result


def _get_new_resolvents(
    clause_one: Clause, literal_one: Literal, given_clause: Clause
) -> Tuple[Clause, ...]:
    resolvents: Tuple[Clause, ...] = ()
    for j, literal_two in enumerate(given_clause.literals):
        if literal_one.negated != literal_two.negated:
            clause_two = Clause(
                given_clause.literals[:j] + given_clause.literals[j + 1 :]
            )
            try:
                resolvents = resolvents + (
                    resolution(
                        clause_one, literal_one, clause_two, literal_two
                    ),
                )
            except NonUnifiableError:
                pass
    return resolvents


def all_possible_resolvents(
    clauses: Tuple[Clause, ...],
    given_clause: Clause,
) -> Tuple[Clause, ...]:
    """
    One of the four basic building blocks of the Given Clause algorithm.

    >>> from tptp_lark_parser.tptp_parser import TPTPParser
    >>> parser = TPTPParser()
    >>> one = parser.parse("cnf(one, axiom, q(X) | p(X)).")[0]
    >>> two = parser.parse("cnf(two, axiom, ~p(c)).")[0]
    >>> tuple(map(
    ...      parser.cnf_parser.pretty_print,
    ...      all_possible_resolvents((one,), two)
    ... ))
    ('cnf(x..., lemma, q(c), inference(resolution, [], [one, two])).',)

    :param clauses: a list of (processed) clauses
    :param given_clause: a new clause which should be combined with all the
        processed ones
    :returns: results of all possible resolvents with each one from
        ``clauses`` and the ``given_clause``
    """
    resolvents: Tuple[Clause, ...] = ()
    for clause in clauses:
        for i, literal_one in enumerate(clause.literals):
            clause_one = Clause(clause.literals[:i] + clause.literals[i + 1 :])
            new_resolvents = _get_new_resolvents(
                clause_one, literal_one, given_clause
            )
            resolvents = resolvents + tuple(
                Clause(
                    literals=resolvent.literals,
                    inference_parents=(clause.label, given_clause.label),
                    inference_rule="resolution",
                )
                for ord_num, resolvent in enumerate(new_resolvents)
            )
    return resolvents
