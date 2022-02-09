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
Unification
============
"""
from typing import Optional, Tuple

from gym_saturation.grammar import Function, Predicate, Proposition, Variable
from gym_saturation.logic_ops.substitution import Substitution
from gym_saturation.logic_ops.utils import is_subproposition


class NonUnifiableError(Exception):
    """exception raised when terms are not unifiable"""


def _get_disagreement(
    one: Proposition, two: Proposition
) -> Tuple[Proposition, ...]:
    """
    find a disagreement set of two first order propositions

    :param one: some proposition
    :param two: some (other but might be the same) proposition
    :returns: either an empty list (if there is no disagreement) or a pair of
        first subterms which disagree
    """
    if isinstance(one, Variable) and isinstance(two, Variable):
        if one.name != two.name:
            return (one, two)
        return ()
    if (
        isinstance(one, Predicate)
        and isinstance(two, Predicate)
        or isinstance(one, Function)
        and isinstance(two, Function)
    ):
        if one.name != two.name or len(one.arguments) != len(two.arguments):
            return (one, two)
        for argument_one, argument_two in zip(one.arguments, two.arguments):
            disagreement = _get_disagreement(argument_one, argument_two)
            if disagreement != ():
                return disagreement
        return ()
    return (one, two)


def _propose_substitution(
    disagreement: Tuple[Proposition, ...],
    propositions: Tuple[Proposition, ...],
) -> Substitution:
    substitution = None
    if isinstance(disagreement[0], Variable) and isinstance(
        disagreement[1], (Variable, Function)
    ):
        substitution = Substitution(disagreement[0], disagreement[1])
    elif isinstance(disagreement[1], Variable) and isinstance(
        disagreement[0], (Variable, Function)
    ):
        substitution = Substitution(disagreement[1], disagreement[0])
    if substitution is None or is_subproposition(
        substitution.variable, substitution.term
    ):
        raise NonUnifiableError((propositions[0], propositions[1]))
    return substitution


def most_general_unifier(
    propositions: Tuple[Proposition, ...],
    substitutions: Optional[Tuple[Substitution, ...]] = None,
) -> Tuple[Substitution, ...]:
    """
    Robinson's 1965 unification algorithm

    >>> most_general_unifier((Variable("X"), Variable("X")))
    ()
    >>> most_general_unifier((Variable("X"), Function("this_is_a_test_case", (Variable("X"),))))
    Traceback (most recent call last):
     ...
    gym_saturation.logic_ops.unification.NonUnifiableError: (Variable(name='X'), Function(name='this_is_a_test_case', arguments=(Variable(name='X'),)))
    >>> most_general_unifier((Predicate("s", (Variable("X"), Variable("Y"), Function("f", (Variable("Z"),)))), Predicate("s", (Variable("U"), Function("this_is_a_test_case", (Variable("V"), Variable("V"))), Variable("V")))))
    (Substitution(variable=Variable(name='X'), term=Variable(name='U')), Substitution(variable=Variable(name='Y'), term=Function(name='this_is_a_test_case', arguments=(Variable(name='V'), Variable(name='V')))), Substitution(variable=Variable(name='V'), term=Function(name='f', arguments=(Variable(name='Z'),))))

    :param propositions: a set of propositions to unify
    :param substitutions: this function is recursive and uses this param to
        accumulate its output
    :returns: a map from variables to functions or other variables (aka
        substitution)
    """
    not_none_substitutions = () if substitutions is None else substitutions
    if len(propositions) == 1:
        return not_none_substitutions
    disagreement = _get_disagreement(propositions[0], propositions[1])
    if disagreement == ():
        return ()
    substitution = _propose_substitution(disagreement, propositions)
    return most_general_unifier(
        tuple(
            set(
                substitution(proposition)  # type: ignore
                for proposition in propositions
            )
        ),
        not_none_substitutions + (substitution,),
    )
