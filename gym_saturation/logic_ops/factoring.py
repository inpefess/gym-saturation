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

from gym_saturation import grammar
from gym_saturation.logic_ops.unification import (
    NonUnifiableError,
    most_general_unifier,
)
from gym_saturation.utils import pickle_copy


def factoring(
    given_clause: grammar.Clause,
    literal_one: grammar.Literal,
    literal_two: grammar.Literal,
) -> grammar.Clause:
    r"""
    positive factoring rule

    .. math:: {\frac{\Gamma\cup\left\{L_1,L_2\right\}}{\left(\Gamma\cup\left\{L_1\right\}\right)\phi}}

    where

    * :math:`\Gamma` and is a clause
    * :math:`L_1` and :math:`L_2` are literals, both without negation
    * :math:`\phi` is a most general unifier of :math:`L_1` and :math:`L_2`

    >>> from gym_saturation.grammar import Predicate, Variable, Function
    >>> factoring(grammar.Clause([grammar.Literal(True, Predicate("q", [Variable("X")]))]), grammar.Literal(False, Predicate("p", [Variable("X")])), grammar.Literal(False, Predicate("p", [Function("this_is_a_test_case", [])]))).literals
    [Literal(negated=True, atom=Predicate(name='q', arguments=[Function(name='this_is_a_test_case', arguments=[])])), Literal(negated=False, atom=Predicate(name='p', arguments=[Function(name='this_is_a_test_case', arguments=[])]))]
    >>> factoring(grammar.Clause([]), grammar.Literal(False, Predicate("f", [])), grammar.Literal(True, Predicate("this_is_a_test_case", [])))
    Traceback (most recent call last):
     ...
    ValueError: factoring is not possible for Literal(negated=False, atom=Predicate(name='f', arguments=[])) and Literal(negated=True, atom=Predicate(name='this_is_a_test_case', arguments=[]))

    :param given_clause: :math:`\Gamma`
    :param literal_one: :math:`L_1`
    :param literal_two: :math:`L_2`
    :returns: a new clause --- the factoring result
    """
    if literal_one.negated or literal_two.negated:
        raise ValueError(
            f"factoring is not possible for {literal_one} and {literal_two}"
        )
    substitutions = most_general_unifier([literal_one.atom, literal_two.atom])
    new_literals = pickle_copy(given_clause.literals + [literal_one])
    result = grammar.Clause(new_literals)
    for substitution in substitutions:
        result = substitution.substitute_in_clause(result)
    return result


def all_possible_factors(
    given_clause: grammar.Clause,
    label_prefix: str,
    starting_label_index: int,
) -> List[grammar.Clause]:
    """
    one of the four basic building blocks of the Given Clause algorithm

    >>> from gym_saturation.grammar import Predicate, Variable, Function
    >>> all_possible_factors(grammar.Clause([grammar.Literal(False, Predicate("this_is_a_test_case", []))]), "inferred_", 0)
    Traceback (most recent call last):
     ...
    ValueError: no label: Clause(literals=[Literal(negated=False, atom=Predicate(name='this_is_a_test_case', arguments=[]))], label=None, inference_parents=None, inference_rule=None, processed=None, birth_step=None)
    >>> all_possible_factors(grammar.Clause([grammar.Literal(False, Predicate("p", [Function("f", [])])), grammar.Literal(False, Predicate("p", [Variable("X")])), grammar.Literal(False, Predicate("q", []))], label="this_is_a_test_case"), "inferred_", 0)
    [Clause(literals=[Literal(negated=False, atom=Predicate(name='q', arguments=[])), Literal(negated=False, atom=Predicate(name='p', arguments=[Function(name='f', arguments=[])]))], label='inferred_0', inference_parents=['this_is_a_test_case'], inference_rule='factoring', processed=None, birth_step=None)]

    :param clauses: a list of (processed) clauses
    :param given_clause: a new clause which should be combined with all the
        processed ones
    :param label_prefix: generated clauses will be labeled with this prefix
    :param starting_label_index: generated clauses will be indexed starting
        with this number
    :returns: results of all possible factors with each one from
        ``clauses`` and the ``given_clause``
    """
    if given_clause.label is None:
        raise ValueError(f"no label: {given_clause}")
    factors: List[grammar.Clause] = []
    for i, literal_one in enumerate(given_clause.literals):
        for j in range(i + 1, len(given_clause.literals)):
            if (
                not literal_one.negated
                and not given_clause.literals[j].negated
            ):
                a_clause = grammar.Clause(
                    given_clause.literals[:i]
                    + given_clause.literals[i + 1 : j]
                    + given_clause.literals[j + 1 :]
                )
                try:
                    factors.append(
                        factoring(
                            a_clause, literal_one, given_clause.literals[j]
                        )
                    )
                except NonUnifiableError:
                    pass
    return [
        grammar.Clause(
            literals=factor.literals,
            inference_parents=[given_clause.label],
            inference_rule="factoring",
            label=label_prefix + str(starting_label_index + ord_num),
        )
        for ord_num, factor in enumerate(factors)
    ]
