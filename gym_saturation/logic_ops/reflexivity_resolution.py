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
Reflexivity Resolution
=======================
"""
from typing import Tuple

from tptp_lark_parser import grammar

from gym_saturation.logic_ops.unification import (
    NonUnifiableError,
    most_general_unifier,
)


def reflexivity_resolution(
    given_clause: grammar.Clause, a_literal: Tuple[grammar.Term, grammar.Term]
) -> grammar.Clause:
    r"""
    Apply reflexivity resolution rule.

    .. _reflexivity_resolution:

    .. math:: \frac{C\vee s\not\approx t}{\sigma\left(C\right)}

    where

    * :math:`C` and is a clause
    * :math:`s` and :math:`t` are terms, :math:`\not\approx` is a negation of equality
    * :math:`\sigma` is a most general unifier of :math:`s` and :math:`t`

    >>> from tptp_lark_parser.grammar import Predicate, Variable, Function
    >>> "this_is_a_test_case", reflexivity_resolution(grammar.Clause((grammar.Literal(True, Predicate(7, (Variable(0),))),)), (Variable(0), Function(0, ()))).literals
    ('this_is_a_test_case', (Literal(negated=True, atom=Predicate(index=7, arguments=(Function(index=0, arguments=()),))),))

    :param given_clause: :math:`C`
    :param a_literal: :math:`s\not\approx t`
    :returns: a new clause --- the reflexivity resolution result
    """
    substitutions = most_general_unifier((a_literal[0], a_literal[1]))
    new_literals = given_clause.literals
    result = grammar.Clause(new_literals)
    for substitution in substitutions:
        result = substitution.substitute_in_clause(result)
    return result


def all_possible_reflexivity_resolvents(
    given_clause: grammar.Clause,
) -> Tuple[grammar.Clause, ...]:
    """
    One of the four basic building blocks of the Given Clause algorithm.

    >>> from tptp_lark_parser.tptp_parser import TPTPParser
    >>> parser = TPTPParser()
    >>> clause = parser.parse("cnf(this_is_a_test_case, axiom, p(X) | ~ X=a | b != a).")[0]
    >>> tuple(map(
    ...     parser.cnf_parser.pretty_print,
    ...     all_possible_reflexivity_resolvents(clause)
    ... ))
    ('cnf(x..., lemma, p(a) | ~b = a, inference(reflexivity_resolution, [], [this_is_a_test_case])).',)

    :param given_clause: a new clause which should be combined with all the
        processed ones
    :returns: results of all possible reflexivity resolvents with each one from
        ``clauses`` and the ``given_clause``
    """
    reflexivity_resolvents: Tuple[grammar.Clause, ...] = ()
    for i, a_literal in enumerate(given_clause.literals):
        if (
            a_literal.negated
            and a_literal.atom.index == grammar.EQUALITY_SYMBOL_ID
            or not a_literal.negated
            and a_literal.atom.index == grammar.INEQUALITY_SYMBOL_ID
        ) and len(a_literal.atom.arguments) == 2:
            a_clause = grammar.Clause(
                given_clause.literals[:i] + given_clause.literals[i + 1 :]
            )
            if a_clause.literals:
                try:
                    reflexivity_resolvents = reflexivity_resolvents + (
                        reflexivity_resolution(
                            a_clause,
                            (
                                a_literal.atom.arguments[0],
                                a_literal.atom.arguments[1],
                            ),
                        ),
                    )
                except NonUnifiableError:
                    pass
    return tuple(
        grammar.Clause(
            literals=reflexivity_resolvent.literals,
            inference_parents=(given_clause.label,),
            inference_rule="reflexivity_resolution",
        )
        for ord_num, reflexivity_resolvent in enumerate(reflexivity_resolvents)
    )
