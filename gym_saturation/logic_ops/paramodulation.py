# Copyright 2021 Boris Shminke

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
Paramodulation
===============
"""
from typing import List, Tuple

from gym_saturation.grammar import Clause, Literal, Term
from gym_saturation.logic_ops.unification import (
    NonUnifiableError,
    most_general_unifier,
)
from gym_saturation.logic_ops.utils import (
    TermSelfReplace,
    proposition_length,
    replace_subterm_by_index,
    subterm_by_index,
)
from gym_saturation.utils import deduplicate, pickle_copy


def paramodulation(
    clause_one: Clause,
    literal_one: Tuple[Term, Term],
    clause_two: Clause,
    literal_two: Literal,
    r_position: int,
) -> Clause:
    r"""
    .. _paramodulation:

    binary paramodulation rule

    .. math:: {\frac{C_1\vee s\approx t,C_2\vee L\left[r\right]}{\sigma\left(L\left[t\right]\vee C_1\vee C_2\right)}}

    where

    * :math:`C_1` and :math:`C_2` are clauses with no common variables
    * :math:`L\left[r\right]` is a literal with a subterm :math:`r`
    * only one instance of :math:`r` in :math:`L\left[r\right]` is considered, even if there are many of them
    * :math:`s` and :math:`t` are terms, :math:`\approx` is a syntactic equality symbol
    * :math:`\sigma` is a most general unifier of :math:`s` and :math:`r`

    >>> from gym_saturation.grammar import Predicate, Variable, Function
    >>> paramodulation(Clause([Literal(False, Predicate("q", [Variable("X")]))]), [Variable("X"), Function("this_is_a_test_case", [])], Clause([Literal(False, Predicate("r", [Variable("X")]))]), Literal(True, Predicate("p", [Function("f", [])])), 1).literals
    [Literal(negated=True, atom=Predicate(name='p', arguments=[Function(name='this_is_a_test_case', arguments=[])])), Literal(negated=False, atom=Predicate(name='q', arguments=[Function(name='f', arguments=[])])), Literal(negated=False, atom=Predicate(name='r', arguments=[Function(name='f', arguments=[])]))]

    :param clause_one: :math:`C_1`
    :param literal_one: :math:`s\approx t`
    :param clause_two: :math:`C_2`
    :param literal_two: :math:`L\left[r\right]`
    :param r_position: index of :math:`r` in the tree of subterms of :math:`L\left[r\right]`
    :returns: a new clause --- the paramodulation result
    """
    substitutions = most_general_unifier(
        [
            literal_one[0],
            subterm_by_index(literal_two.atom, r_position),
        ]
    )
    new_atom = pickle_copy(literal_two.atom)
    replace_subterm_by_index(
        new_atom,
        r_position,
        literal_one[1],
    )
    new_literals = deduplicate(
        [Literal(literal_two.negated, new_atom)]
        + pickle_copy(clause_one.literals)
        + pickle_copy(clause_two.literals)
    )
    result = Clause(new_literals)
    for substitution in substitutions:
        result = substitution.substitute_in_clause(result)
    return result


def _equality_symmetry_paramodulation(
    clause_one, literal_one, clause_two, literal_two, k
):
    paramodulants = []
    try:
        paramodulants.append(
            paramodulation(
                clause_one,
                (
                    literal_one.atom.arguments[0],
                    literal_one.atom.arguments[1],
                ),
                clause_two,
                literal_two,
                k,
            )
        )
    except (NonUnifiableError, TermSelfReplace):
        pass
    try:
        paramodulants.append(
            paramodulation(
                clause_one,
                (
                    literal_one.atom.arguments[1],
                    literal_one.atom.arguments[0],
                ),
                clause_two,
                literal_two,
                k,
            )
        )
    except (NonUnifiableError, TermSelfReplace):
        pass
    return paramodulants


def all_paramodulants_from_clause(
    clause_one: Clause,
    literal_one: Literal,
    clause_two: Clause,
    literal_two: Literal,
) -> List[Clause]:
    r"""
    applies ``paramodulation`` with varying ``r_position`` argument
    also varies for equality symmetry

    :param clause_one: :math:`C_1`
    :param literal_one: :math:`s\approx t`
    :param clause_two: :math:`C_2`
    :param literal_two: :math:`L\left[r\right]`
    :returns: a list of paramodulants for all possible values of ``r_position``
    """
    if (
        literal_one.atom.name != "="
        or literal_one.negated
        or len(literal_one.atom.arguments) != 2
    ):
        raise ValueError(f"expected equality, but got {literal_one}")
    if literal_one.atom.arguments[0] == literal_one.atom.arguments[1]:
        return []
    literal_two_len = proposition_length(literal_two.atom)
    paramodulants = []
    for k in range(1, literal_two_len):
        paramodulants.extend(
            _equality_symmetry_paramodulation(
                clause_one, literal_one, clause_two, literal_two, k
            )
        )
    return paramodulants


def _get_new_paramodulants(
    clause_one: Clause, literal_one: Literal, given_clause: Clause
) -> List[Clause]:
    paramodulants = []
    for j, literal_two in enumerate(given_clause.literals):
        clause_two = Clause(
            given_clause.literals[:j] + given_clause.literals[j + 1 :]
        )
        if not literal_one.negated and literal_one.atom.name == "=":
            paramodulants.extend(
                all_paramodulants_from_clause(
                    clause_one, literal_one, clause_two, literal_two
                )
            )
        if not literal_two.negated and literal_two.atom.name == "=":
            # pylint: disable=arguments-out-of-order
            paramodulants.extend(
                all_paramodulants_from_clause(
                    clause_two, literal_two, clause_one, literal_one
                )
            )
    return paramodulants


def all_paramodulants_from_list(
    clauses: List[Clause],
    given_clause: Clause,
) -> List[Clause]:
    """
    one of the four basic building block of the Given Clause algorithm

    >>> from gym_saturation.grammar import Literal, Function, Predicate
    >>> all_paramodulants_from_list([Clause([Literal(False, Predicate("=", [Function("this_is_a_test_case", [])]))], "one")], Clause([Literal(True, Predicate("p", []))], "two"))
    Traceback (most recent call last):
     ...
    ValueError: expected equality, but got Literal(negated=False, atom=Predicate(name='=', arguments=[Function(name='this_is_a_test_case', arguments=[])]))
    >>> from gym_saturation.parsing.tptp_parser import TPTPParser
    >>> parser = TPTPParser()
    >>> one = parser.parse("cnf(one, axiom, a=b | X=X).", "")[0]
    >>> two = parser.parse("cnf(two, axiom, b=c).", "")[0]
    >>> res = all_paramodulants_from_list([one], two)
    >>> dedup = map(Clause, deduplicate([clause.literals for clause in res]))
    >>> print("\\n".join(map(str, dedup)))  # doctest: +ELLIPSIS
    cnf(..., hypothesis, a = c | X = X).
    cnf(..., hypothesis, c = b | a = b).
    cnf(..., hypothesis, b = c | a = b).

    :param clauses: a list of (processed) clauses
    :param given_clause: a new clause which should be combined with all the
        processed ones
    :param label_prefix: generated clauses will be labeled with this prefix
    :param label_index_base: generated clauses will be indexed starting
        with this number
    :returns: results of all possible paramodulants with each one from
        ``clauses`` and the ``given_clause``
    """
    paramodulants: List[Clause] = []
    for other_clause in clauses:
        for i, literal_one in enumerate(other_clause.literals):
            clause_one = Clause(
                other_clause.literals[:i] + other_clause.literals[i + 1 :]
            )
            new_paramodulants = _get_new_paramodulants(
                clause_one, literal_one, given_clause
            )
            paramodulants.extend(
                [
                    Clause(
                        literals=paramodulant.literals,
                        inference_parents=[
                            other_clause.label,
                            given_clause.label,
                        ],
                        inference_rule="paramodulation",
                    )
                    for ord_num, paramodulant in enumerate(new_paramodulants)
                ]
            )
    return paramodulants
