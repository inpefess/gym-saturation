"""
Copyright 2021 Boris Shminke

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Tuple

from gym_saturation.grammar import Clause, Literal
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
    """
    standard first-order resolution rule

    :math:`{\\frac  {\\Gamma _{1}\\cup \\left\\{L_{1}\\right\\}\\,\\,\\,\\,\\Gamma _{2}\\cup \\left\\{L_{2}\\right\\}}{(\\Gamma _{1}\\cup \\Gamma _{2})\\phi }}\\phi`

    where :math:`\\phi` is a most general unifier of :math:`L_{1}` and :math:`\\overline {L_{2}}`, and :math:`\\Gamma _{1}` and :math:`\\Gamma _{2}` have no common variables.

    >>> from gym_saturation.grammar import Predicate, Variable, Function
    >>> resolution(Clause([Literal(False, Predicate("q", [Variable("X")]))]), Literal(False, Predicate("p", [Variable("X")])), Clause([Literal(False, Predicate("r", [Variable("X")]))]), Literal(True, Predicate("p", [Function("this_is_a_test_case", [])]))).literals
    [Literal(negated=False, atom=Predicate(name='q', arguments=[Function(name='this_is_a_test_case', arguments=[])])), Literal(negated=False, atom=Predicate(name='r', arguments=[Function(name='this_is_a_test_case', arguments=[])]))]
    >>> resolution(Clause([]), Literal(False, Predicate("f", [])), Clause([]), Literal(False, Predicate("this_is_a_test_case", [])))
    Traceback (most recent call last):
     ...
    ValueError: resolution is not possible for Literal(negated=False, atom=Predicate(name='f', arguments=[])) and Literal(negated=False, atom=Predicate(name='this_is_a_test_case', arguments=[]))

    :param clause_one: :math:`\\Gamma_{1}`
    :param literal_one: :math:`L_{1}`
    :param clause_two: :math:`\\Gamma_{2}`
    :param literal_two: :math:`L_{2}`
    :returns: a new clause --- the resolution result
    """
    if literal_one.negated == literal_two.negated:
        raise ValueError(
            f"resolution is not possible for {literal_one} and {literal_two}"
        )
    substitutions = most_general_unifier([literal_one.atom, literal_two.atom])
    new_literals = clause_one.literals
    for literal in clause_two.literals:
        if literal not in new_literals:
            new_literals.append(literal)
    result = Clause(new_literals)
    for substitution in substitutions:
        result = substitution.substitute_in_clause(result)
    return result


def _multi_resolution_init(
    given_clause: Clause, clauses: List[Clause], starting_label_index: int
) -> Tuple[List[Literal], List[Literal], List[Clause], int]:
    for clause in clauses + [given_clause]:
        if clause.label is None:
            raise ValueError(f"clauses should be labeled: {clause}")
    negative_literals = [
        literal for literal in given_clause.literals if literal.negated
    ]
    positive_literals = [
        literal for literal in given_clause.literals if not literal.negated
    ]
    resolutions: List[Clause] = list()
    new_clause_index = starting_label_index
    return negative_literals, positive_literals, resolutions, new_clause_index


def all_possible_resolutions(
    clauses: List[Clause],
    given_clause: Clause,
    label_prefix: str,
    starting_label_index: int,
) -> List[Clause]:
    """
    basic building block of the Given Clause algorithm

    >>> from gym_saturation.grammar import Predicate, Variable, Function
    >>> all_possible_resolutions([Clause([Literal(False, Predicate("this_is_a_test_case", []))])], Clause([]), "inferred_", 0)
    Traceback (most recent call last):
     ...
    ValueError: clauses should be labeled: Clause(literals=[Literal(negated=False, atom=Predicate(name='this_is_a_test_case', arguments=[]))], label=None, inference_parents=None, processed=None, birth_step=None)
    >>> all_possible_resolutions([Clause([Literal(False, Predicate("q", [Variable("X")])), Literal(False, Predicate("p", [Variable("X")]))], label="input1")], Clause([Literal(True, Predicate("p", [Function("this_is_a_test_case", [])]))], label="input2"), "inferred_", 0)
    [Clause(literals=[Literal(negated=False, atom=Predicate(name='q', arguments=[Function(name='this_is_a_test_case', arguments=[])]))], label='inferred_0', inference_parents=['input1', 'input2'], processed=None, birth_step=None)]

    :param clauses: a list of (processed) clauses
    :param given_clause: a new clause which should be combined with all the
        processed ones
    :param label_prefix: generated clauses will be labeled with this prefix
    :param starting_label_index: generated clauses will be indexed starting
        with this number
    :returns: results of all possible resolutions with each one from
        ``clauses`` and the ``given_clause``
    """
    (
        negative_literals,
        positive_literals,
        resolutions,
        new_clause_index,
    ) = _multi_resolution_init(given_clause, clauses, starting_label_index)
    for clause in clauses:
        for i, _ in enumerate(clause.literals):
            other_literals = (
                positive_literals
                if clause.literals[i].negated
                else negative_literals
            )
            for j, _ in enumerate(other_literals):
                try:
                    new_literals = resolution(
                        Clause(clause.literals[:i] + clause.literals[i + 1 :]),
                        clause.literals[i],
                        Clause(
                            given_clause.literals[:j]
                            + given_clause.literals[j + 1 :]
                        ),
                        other_literals[j],
                    ).literals
                except NonUnifiableError:
                    continue
                new_clause = Clause(
                    new_literals,
                    f"{label_prefix}{new_clause_index}",
                    [
                        # check for non empty labels is done in
                        # ``_multi_resolution_init``
                        clause.label,  # type: ignore
                        given_clause.label,  # type: ignore
                    ],
                )
                new_clause_index += 1
                resolutions.append(new_clause)
    return resolutions
