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

from gym_saturation import grammar, utils
from gym_saturation.logic_ops.unification import (
    NonUnifiableError,
    most_general_unifier,
)


def reflexivity_resolution(
    given_clause: grammar.Clause, a_literal: Tuple[grammar.Term, grammar.Term]
) -> grammar.Clause:
    r"""
    reflexivity resolution rule

    .. math:: \frac{C\vee s\not\approx t}{\sigma\left(C\right)}

    where

    * :math:`C` and is a clause
    * :math:`s` and :math:`t` are terms, :math:`\not\approx` is a negation of equality
    * :math:`\sigma` is a most general unifier of :math:`s` and :math:`t`

    >>> from gym_saturation.grammar import Predicate, Variable, Function
    >>> reflexivity_resolution(grammar.Clause([grammar.Literal(True, Predicate("this_is_a_test_case", [Variable("X")]))]), [Variable("X"), Function("f", [])]).literals
    [Literal(negated=True, atom=Predicate(name='this_is_a_test_case', arguments=[Function(name='f', arguments=[])]))]

    :param given_clause: :math:`C`
    :param a_literal: :math:`s\not\approx t`
    :returns: a new clause --- the reflexivity resolution result
    """
    substitutions = most_general_unifier([a_literal[0], a_literal[1]])
    new_literals = utils.pickle_copy(given_clause.literals)
    result = grammar.Clause(new_literals)
    for substitution in substitutions:
        result = substitution.substitute_in_clause(result)
    return result


def all_possible_reflexivity_resolvents(
    given_clause: grammar.Clause,
    label_prefix: str,
    label_index_base: int,
) -> List[grammar.Clause]:
    """
    one of the four basic building blocks of the Given Clause algorithm

    >>> from gym_saturation.grammar import Predicate, Variable, Function
    >>> all_possible_reflexivity_resolvents(grammar.Clause([grammar.Literal(False, Predicate("this_is_a_test_case", []))]), "inferred_", 1)
    Traceback (most recent call last):
     ...
    ValueError: no label: Clause(literals=[Literal(negated=False, atom=Predicate(name='this_is_a_test_case', arguments=[]))], label=None, inference_parents=None, inference_rule=None, processed=None, birth_step=None)
    >>> all_possible_reflexivity_resolvents(grammar.Clause([grammar.Literal(False, Predicate("p", [Variable("X")])), grammar.Literal(False, Predicate("=", [Variable("X"), Function("g", [])])), grammar.Literal(True, Predicate("!=", [Function("f", []), Function("g", [])]))], label="this_is_a_test_case"), "inferred_", 0)
    [Clause(literals=[Literal(negated=False, atom=Predicate(name='p', arguments=[Function(name='g', arguments=[])])), Literal(negated=True, atom=Predicate(name='!=', arguments=[Function(name='f', arguments=[]), Function(name='g', arguments=[])]))], label='inferred_0', inference_parents=['this_is_a_test_case'], inference_rule='reflexivity_resolution', processed=None, birth_step=None)]

    :param given_clause: a new clause which should be combined with all the
        processed ones
    :param label_prefix: generated clauses will be labeled with this prefix
    :param label_index_base: generated clauses will be indexed starting
        with this number
    :returns: results of all possible reflexivity resolvents with each one from
        ``clauses`` and the ``given_clause``
    """
    if given_clause.label is None:
        raise ValueError(f"no label: {given_clause}")
    reflexivity_resolvents: List[grammar.Clause] = []
    for i, a_literal in enumerate(given_clause.literals):
        if (
            not a_literal.negated
            and a_literal.atom.name == "="
            or a_literal.negated
            and a_literal.atom.name == "!="
        ) and len(a_literal.atom.arguments) == 2:
            a_clause = grammar.Clause(
                given_clause.literals[:i] + given_clause.literals[i + 1 :]
            )
            try:
                reflexivity_resolvents.append(
                    reflexivity_resolution(
                        a_clause,
                        (
                            a_literal.atom.arguments[0],
                            a_literal.atom.arguments[1],
                        ),
                    )
                )
            except NonUnifiableError:
                pass
    return [
        grammar.Clause(
            literals=reflexivity_resolvent.literals,
            inference_parents=[given_clause.label],
            inference_rule="reflexivity_resolution",
            label=label_prefix + str(label_index_base + ord_num),
        )
        for ord_num, reflexivity_resolvent in enumerate(reflexivity_resolvents)
    ]
