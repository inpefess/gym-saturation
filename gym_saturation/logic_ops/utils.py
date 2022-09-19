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
Logic Operations Utility Functions
===================================
"""
import dataclasses
from itertools import chain
from operator import attrgetter
from typing import Any, Dict, Tuple, Union

from tptp_lark_parser.grammar import (
    EQUALITY_SYMBOL_ID,
    Clause,
    Function,
    Predicate,
    Proposition,
    Term,
    Variable,
)

from gym_saturation.logic_ops.substitution import Substitution


class WrongRefutationProofError(Exception):
    """Exception raised when proof is requested but not found yet."""


def is_subproposition(one: Proposition, two: Proposition) -> bool:
    """
    Check whether proposition ``one`` is part of a proposition ``two``.

    :param one: presumably shorter proposition which is probably a part
    :param two: presumably longer proposition which is probably the whole
    :returns: whether ``one`` is a part of ``two``
    """
    if isinstance(two, Variable):
        return one == two
    if isinstance(two, (Function, Predicate)):
        for argument in two.arguments:
            if is_subproposition(one, argument):
                return True
    return False


def get_variable_list(
    clause: Union[Clause, Proposition]
) -> Tuple[Variable, ...]:
    """
    Find all variables present in a clause.

    >>> from tptp_lark_parser.grammar import Literal
    >>> get_variable_list(Clause((Literal(False,
    ...     Predicate(0, (Function(0, (Variable(0), Variable(0))),))
    ... ),)))
    (Variable(index=0), Variable(index=0))

    :param clause: a clause
    :returns: a list (with repetitions) of variables from there clause
    """
    variable_list: Tuple[Variable, ...] = ()
    if isinstance(clause, Clause):
        for literal in clause.literals:
            for term in literal.atom.arguments:
                variable_list = variable_list + get_variable_list(term)
    elif isinstance(clause, Function):
        for term in clause.arguments:
            variable_list = variable_list + get_variable_list(term)
    elif isinstance(clause, Variable):
        variable_list = variable_list + (clause,)
    return variable_list


def _shift_variables(
    clauses: Dict[str, Clause], variable_list: Tuple[Variable, ...], shift: int
) -> Dict[str, Clause]:
    new_clauses: Dict[str, Clause] = {}
    for label, clause in clauses.items():
        new_clause = clause
        for i, variable in enumerate(variable_list):
            new_clause = Substitution(
                variable, Variable(shift + i)
            ).substitute_in_clause(new_clause)
        new_clauses[label] = new_clause
    return new_clauses


def reindex_variables(clauses: Dict[str, Clause]) -> Dict[str, Clause]:
    """
    Rename variables so that each clause has its unique set of variables.

    >>> from tptp_lark_parser.grammar import Literal
    >>> clause = Clause((Literal(False,
    ...     Predicate(0, (Variable(2), Variable(1), Variable(0)))
    ... ),))
    >>> sorted(map(
    ...     attrgetter("index"),
    ...     reindex_variables(
    ...         {"this_is_a_test_case": clause}
    ...     )["this_is_a_test_case"].literals[0].atom.arguments
    ... ))
    [0, 1, 2]
    >>> clause = Clause((Literal(False,
    ...     Predicate(0, (Variable(5), Variable(10), Variable(5)))
    ... ),))
    >>> sorted(map(
    ...     attrgetter("index"),
    ...     reindex_variables(
    ...         {"this_is_a_test_case": clause}
    ...     )["this_is_a_test_case"].literals[0].atom.arguments
    ... ))
    [0, 1, 1]

    :param clauses: a map of clause labels to clauses
    :returns: the list of clauses with renamed variables
    """
    variable_list = _flat_list(tuple(map(get_variable_list, clauses.values())))
    shift = max(
        len(variable_list),
        1 + max(map(attrgetter("index"), variable_list), default=-1),
    )
    new_clauses = _shift_variables(clauses, variable_list, shift)
    variable_list = _flat_list(
        tuple(map(get_variable_list, new_clauses.values()))
    )
    new_clauses = _shift_variables(new_clauses, variable_list, 0)
    return new_clauses


def is_tautology(clause: Clause) -> bool:
    """
    Check whether there are two literals (negated and not) with the same atom.

    >>> from tptp_lark_parser.grammar import Literal
    >>> is_tautology(Clause((Literal(False, Predicate(7, ())),)))
    False
    >>> is_tautology(Clause(
    ...     (Literal(False, Predicate(7, ())), Literal(True, Predicate(7, ())))
    ... ))
    True
    >>> is_tautology(Clause(
    ...     (Literal(False, Predicate(1, (Variable(0), Variable(0)))),)
    ... ))
    True

    :param clause: a clause to check
    :returns: whether the clause is a primitive tautology or not
    """
    for i, literal in enumerate(clause.literals):
        for j, another_literal in enumerate(clause.literals):
            if (
                i != j
                and literal.negated != another_literal.negated
                and literal.atom == another_literal.atom
            ):
                return True
        if literal.atom.index == EQUALITY_SYMBOL_ID and (
            literal.atom.arguments[0] == literal.atom.arguments[1]
        ):
            return True
    return False


def clause_length(clause: dict) -> int:
    """
    Find the length of arguments of each predicate.

    Negation adds one to each literal.

    :param clause: a clause in JSON representation
    :return: structural length of a clause

    >>> from tptp_lark_parser.grammar import Literal
    >>> import orjson
    >>> clause_length(orjson.loads(orjson.dumps(
    ...     Clause((Literal(True, Predicate("p", (Function("this_is_a_test_case", ()),))),))
    ... )))
    3
    """
    length = 0
    if isinstance(clause, dict):
        for key, value in clause.items():
            if key in {"negated", "index"}:
                length += 1
            if isinstance(value, dict):
                length += clause_length(value)
            if isinstance(value, (list, tuple)):
                for item in value:
                    length += clause_length(item)
    return length


def proposition_length(proposition: Proposition) -> int:
    """
    Find the number of functional, predicate and variable symbols.

    :param proposition: a function, a predicate or a variable
    :return: structural length of a proposition

    >>> proposition_length(Predicate(7, (Function(0, (Variable(0),)),)))
    3
    """
    length = 0
    if isinstance(proposition, Variable):
        return 1
    for subterm in proposition.arguments:
        length += proposition_length(subterm)
    return 1 + length


def clause_in_a_list(clause: Clause, clauses: Tuple[Clause, ...]) -> bool:
    """
    Check whether a clause is in a list.

    >>> clause_in_a_list(Clause((), label="one"), (Clause((), label="two"),))
    True

    :param clause: some clause
    :param clauses: a list of clauses
    :returns: whether in the list there is a clause with a literals set to a
        given clause
    """
    return clause.literals in set(map(lambda clause: clause.literals, clauses))


class NoSubtermFound(Exception):
    """Sometimes a sub-term index is larger than term length."""


def subterm_by_index(atom: Proposition, index: int) -> Term:
    """
    Extract a sub-term using depth-first search.

    >>> atom = Predicate(7, (
    ...     Function(0, (Variable(0),)), Function(1, (Variable(1),))
    ... ))
    >>> subterm_by_index(atom, 0)
    Traceback (most recent call last):
     ...
    ValueError: sub-term with index 0 exists only for terms, but got: ...
    >>> subterm_by_index(atom, 1) == atom.arguments[0]
    True
    >>> subterm_by_index(atom, 2) == atom.arguments[0].arguments[0]
    True
    >>> subterm_by_index(atom, 4) == atom.arguments[1].arguments[0]
    True

    :param atom: a predicate or a term
    :param index: an index of a desired sub-term
    :returns: a sub-term
    :raises ValueError: when trying to get a term with index 0 of a predicate
    :raises NoSubtermFound: if sub-term with a given index doesn't exist
    """
    if index == 0:
        if isinstance(atom, (Function, Variable)):
            return atom
        raise ValueError(
            f"sub-term with index 0 exists only for terms, but got: {atom}"
        )
    subterm_length = 1
    if not isinstance(atom, Variable):
        for argument in atom.arguments:
            try:
                return subterm_by_index(argument, index - subterm_length)
            except NoSubtermFound as error:
                subterm_length += error.args[0]
    raise NoSubtermFound(subterm_length)


class CantReplaceTheWholeTerm(Exception):
    """An exception raised when trying to replace a sub-term with index 0."""


class TermSelfReplace(Exception):
    """An exception raised when trying to replace a sub-term with itself."""


def _replace_if_not_the_same(old_term: Term, new_term: Term) -> Term:
    if old_term == new_term:
        raise TermSelfReplace
    return new_term


def replace_subterm_by_index(
    atom: Proposition, index: int, term: Term
) -> Proposition:
    """
    Replace a sub-term with a given index (depth-first search) by a new term.

    >>> atom = Predicate(7, (
    ...     Function(0, (Variable(0),)), Function(1, (Variable(2),))
    ... ))
    >>> replace_subterm_by_index(atom, 0, Variable(3))
    Traceback (most recent call last):
     ...
    gym_saturation.logic_ops.utils.NoSubtermFound: 5
    >>> "this_is_a_test_case", replace_subterm_by_index(atom, 4, Function(2, (Variable(3),)))
    ('this_is_a_test_case', Predicate(index=7, arguments=(Function(index=0, arguments=(Variable(index=0),)), Function(index=1, arguments=(Function(index=2, arguments=(Variable(index=3),)),)))))
    >>> replace_subterm_by_index(Predicate(7, (Variable(0),)), 1, Variable(0))
    Traceback (most recent call last):
     ...
    gym_saturation.logic_ops.utils.TermSelfReplace

    :param atom: a predicate or a term
    :param index: an index of a sub-term to replace, must be greater than 0
    :param term: replacement term for a given index
    :returns:
    :raises NoSubtermFound: if sub-term with a given index doesn't exist
    """
    subterm_length = 1
    if not isinstance(atom, Variable):
        for i, argument in enumerate(atom.arguments):
            if index == subterm_length:
                return dataclasses.replace(
                    atom,
                    arguments=atom.arguments[:i]
                    + (_replace_if_not_the_same(argument, term),)
                    + atom.arguments[i + 1 :],
                )
            try:
                return dataclasses.replace(
                    atom,
                    arguments=atom.arguments[:i]
                    + (
                        replace_subterm_by_index(
                            argument, index - subterm_length, term
                        ),
                    )
                    + atom.arguments[i + 1 :],
                )
            except NoSubtermFound as error:
                subterm_length += error.args[0]
    raise NoSubtermFound(subterm_length)


def _flat_list(list_of_lists: Tuple[Tuple[Any, ...], ...]) -> Tuple[Any, ...]:
    return tuple(set(chain(*list_of_lists)))


def reduce_to_proof(clauses: Dict[str, Clause]) -> Tuple[Clause, ...]:
    """
    Leave only clauses belonging to the refutation proof.

    >>> reduce_to_proof({
    ...     "one": Clause((), label="one"), "two": Clause((), label="two")
    ... })
    Traceback (most recent call last):
     ...
    gym_saturation.logic_ops.utils.WrongRefutationProofError
    >>> state = {"one": Clause((), label="one")}
    >>> reduce_to_proof(state) == (Clause((), label="one"), )
    True

    :param clauses: a map of clause labels to clauses
    :returns: the reduced list of clauses
    :raises WrongRefutationProofError: if there is no complete refutation proof
        in a given proof state
    """
    empty_clauses = tuple(
        clause for clause in clauses.values() if clause.literals == tuple()
    )
    if len(empty_clauses) == 1:
        reduced: Tuple[Clause, ...] = ()
        new_reduced: Tuple[Clause, ...] = (empty_clauses[0],)
        while len(new_reduced) > 0:
            reduced += tuple(
                clause for clause in new_reduced if clause not in reduced
            )
            new_reduced = tuple(
                clauses[label]
                for label in tuple(
                    reversed(
                        sorted(
                            _flat_list(
                                tuple(
                                    (
                                        ()
                                        if clause.inference_parents is None
                                        else clause.inference_parents
                                    )
                                    for clause in new_reduced
                                )
                            )
                        )
                    )
                )
            )
        return reduced
    raise WrongRefutationProofError
