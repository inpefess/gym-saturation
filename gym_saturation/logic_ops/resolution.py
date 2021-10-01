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
from typing import List

from gym_saturation.grammar import Clause, Literal
from gym_saturation.logic_ops.unification import (
    NonUnifiableError,
    most_general_unifier,
)
from gym_saturation.utils import pickle_copy


def resolution(
    clause_one: Clause,
    literal_one: Literal,
    clause_two: Clause,
    literal_two: Literal,
) -> Clause:
    r"""
    standard first-order resolution rule

    .. math:: {\frac{C_1\vee L_1,C_2\vee L_2}{\sigma\left(C_1\vee C_2\right)}}

    where

    * :math:`C_1` and :math:`C_2` are clauses with no common variables
    * :math:`L_1` and :math:`L_2` are literals (one negated and one not)
    * :math:`\sigma` is a most general unifier of atoms from :math:`L_1` and :math:`L_2`

    >>> from gym_saturation.grammar import Predicate, Variable, Function
    >>> resolution(Clause([Literal(False, Predicate("q", [Variable("X")]))]), Literal(False, Predicate("p", [Variable("X")])), Clause([Literal(False, Predicate("r", [Variable("X")]))]), Literal(True, Predicate("p", [Function("this_is_a_test_case", [])]))).literals
    [Literal(negated=False, atom=Predicate(name='q', arguments=[Function(name='this_is_a_test_case', arguments=[])])), Literal(negated=False, atom=Predicate(name='r', arguments=[Function(name='this_is_a_test_case', arguments=[])]))]
    >>> resolution(Clause([]), Literal(False, Predicate("f", [])), Clause([]), Literal(False, Predicate("this_is_a_test_case", [])))
    Traceback (most recent call last):
     ...
    ValueError: resolution is not possible for Literal(negated=False, atom=Predicate(name='f', arguments=[])) and Literal(negated=False, atom=Predicate(name='this_is_a_test_case', arguments=[]))

    :param clause_one: :math:`C_1`
    :param literal_one: :math:`L_1`
    :param clause_two: :math:`C_2`
    :param literal_two: :math:`L_2`
    :returns: a new clause --- the resolution result
    """
    if literal_one.negated == literal_two.negated:
        raise ValueError(
            f"resolution is not possible for {literal_one} and {literal_two}"
        )
    substitutions = most_general_unifier([literal_one.atom, literal_two.atom])
    new_literals = pickle_copy(clause_one.literals)
    for literal in clause_two.literals:
        if literal not in new_literals:
            new_literals.append(pickle_copy(literal))
    result = Clause(new_literals)
    for substitution in substitutions:
        result = substitution.substitute_in_clause(result)
    return result


def _get_new_resolvents(
    clause_one: Clause, literal_one: Literal, given_clause: Clause
) -> List[Clause]:
    resolvents: List[Clause] = []
    for j, literal_two in enumerate(given_clause.literals):
        if literal_one.negated != literal_two.negated:
            clause_two = Clause(
                given_clause.literals[:j] + given_clause.literals[j + 1 :]
            )
            try:
                resolvents.append(
                    resolution(
                        clause_one, literal_one, clause_two, literal_two
                    )
                )
            except NonUnifiableError:
                pass
    return resolvents


def all_possible_resolvents(
    clauses: List[Clause],
    given_clause: Clause,
    label_prefix: str,
    starting_label_index: int,
) -> List[Clause]:
    """
    one of the four basic building blocks of the Given Clause algorithm

    >>> from gym_saturation.grammar import Predicate, Variable, Function
    >>> all_possible_resolvents([Clause([])], Clause([Literal(False, Predicate("this_is_a_test_case", []))]), "inferred_", 0)
    Traceback (most recent call last):
     ...
    ValueError: no label: Clause(literals=[Literal(negated=False, atom=Predicate(name='this_is_a_test_case', arguments=[]))], label=None, inference_parents=None, inference_rule=None, processed=None, birth_step=None)
    >>> all_possible_resolvents([Clause([Literal(False, Predicate("this_is_a_test_case", []))])], Clause([], "one"), "inferred_", 0)
    Traceback (most recent call last):
     ...
    ValueError: no label: Clause(literals=[Literal(negated=False, atom=Predicate(name='this_is_a_test_case', arguments=[]))], label=None, inference_parents=None, inference_rule=None, processed=None, birth_step=None)
    >>> all_possible_resolvents([Clause([Literal(False, Predicate("q", [Variable("X")])), Literal(False, Predicate("p", [Variable("X")]))], label="input1")], Clause([Literal(True, Predicate("p", [Function("this_is_a_test_case", [])]))], label="input2"), "inferred_", 0)
    [Clause(literals=[Literal(negated=False, atom=Predicate(name='q', arguments=[Function(name='this_is_a_test_case', arguments=[])]))], label='inferred_0', inference_parents=['input1', 'input2'], inference_rule='resolution', processed=None, birth_step=None)]

    :param clauses: a list of (processed) clauses
    :param given_clause: a new clause which should be combined with all the
        processed ones
    :param label_prefix: generated clauses will be labeled with this prefix
    :param starting_label_index: generated clauses will be indexed starting
        with this number
    :returns: results of all possible resolvents with each one from
        ``clauses`` and the ``given_clause``
    """
    if given_clause.label is None:
        raise ValueError(f"no label: {given_clause}")
    resolvents: List[Clause] = []
    for clause in clauses:
        for i, literal_one in enumerate(clause.literals):
            clause_one = Clause(clause.literals[:i] + clause.literals[i + 1 :])
            if clause.label is None:
                raise ValueError(f"no label: {clause}")
            new_resolvents = _get_new_resolvents(
                clause_one, literal_one, given_clause
            )
            resolvents.extend(
                [
                    Clause(
                        literals=resolvent.literals,
                        inference_parents=[clause.label, given_clause.label],
                        inference_rule="resolution",
                        label=label_prefix
                        + str(
                            starting_label_index + len(resolvents) + ord_num
                        ),
                    )
                    for ord_num, resolvent in enumerate(new_resolvents)
                ]
            )
    return resolvents
