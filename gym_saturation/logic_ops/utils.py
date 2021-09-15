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
from itertools import chain
from typing import List, Union

from gym_saturation.grammar import (
    Clause,
    Function,
    Predicate,
    Proposition,
    Term,
    Variable,
)
from gym_saturation.logic_ops.substitution import Substitution
from gym_saturation.utils import deduplicate, pickle_copy


def is_subproposition(one: Proposition, two: Proposition) -> bool:
    """
    check whether proposition ``one`` is part of a proposition ``two``

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


def get_variable_list(clause: Union[Clause, Proposition]) -> List[Variable]:
    """
    find all variables present in a clause

    >>> from gym_saturation.grammar import Literal
    >>> get_variable_list(Clause([Literal(False, Predicate("this_is_a_test_case", [Function("f", [Variable("X"), Variable("X")])]))]))
    [Variable(name='X'), Variable(name='X')]

    :param clause: a clause
    :returns: a list (with repetitions) of variables from there clause
    """
    variable_list = []
    if isinstance(clause, Clause):
        for literal in clause.literals:
            for term in literal.atom.arguments:
                variable_list.extend(get_variable_list(term))
    elif isinstance(clause, Function):
        for term in clause.arguments:
            variable_list.extend(get_variable_list(term))
    elif isinstance(clause, Variable):
        variable_list.append(clause)
    return variable_list


def reindex_variables(clauses: List[Clause], prefix: str) -> List[Clause]:
    """
    rename variables in a list of clauses so that each clause has its unique
    set of variables

    :param clauses: a list of clauses
    :param prefix: new variables will be named ``prefix[order_num]``
    :returns: the list of clauses with renamed variables
    """
    variable_count = 0
    new_clauses = []
    for clause in clauses:
        new_clause = clause
        variable_list = deduplicate(get_variable_list(clause))
        new_variables_count = len(variable_list)
        for i in range(new_variables_count):
            new_clause = Substitution(
                variable_list[i], Variable(f"{prefix}{i + variable_count}")
            ).substitute_in_clause(new_clause)
        variable_count += new_variables_count
        new_clauses.append(new_clause)
    return new_clauses


def is_tautology(clause: Clause) -> bool:
    """
    check whether there are two literals, one negated and the other not, with
    the same atom

    >>> from gym_saturation.grammar import Literal
    >>> is_tautology(Clause([Literal(False, Predicate("this_is_a_test_case", []))]))
    False
    >>> is_tautology(Clause([Literal(False, Predicate("this_is_a_test_case", [])), Literal(True, Predicate("this_is_a_test_case", []))]))
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
    return False


def clause_length(clause: Clause) -> int:
    """
    total length of arguments of each predicate.
    Negation adds one to each literal

    :param clause: a clause
    :return: sctructural length of a clause

    >>> from gym_saturation.grammar import Literal
    >>> clause_length(Clause([Literal(True, Predicate("p", [Function("this_is_a_test_case", [])]))]))
    3
    """
    length = 0
    for literal in clause.literals:
        if literal.negated:
            length += 1
        for term in literal.atom.arguments:
            length += proposition_length(term)
        length += 1
    return length


def proposition_length(proposition: Proposition) -> int:
    """
    total number of functional, predicate and variable symbols

    :param proposition: a function, a predicate or a variable
    :return: sctructural length of a proposition

    >>> proposition_length(Predicate("p", [Function("f", [Variable("X")])]))
    3
    """
    length = 0
    if isinstance(proposition, Variable):
        return 1
    for subterm in proposition.arguments:
        length += proposition_length(subterm)
    return 1 + length


def clause_in_a_list(clause: Clause, clauses: List[Clause]) -> bool:
    """

    >>> clause_in_a_list(Clause([], label="one"), [Clause([], label="two")])
    True

    :param clause: some clause
    :param clauses: a list of clauses
    :returns: whether in the list there is a clause with a literals set to a
        given clause
    """
    for a_clause in clauses:
        if a_clause.literals == clause.literals:
            return True
    return False


class NoSubtermFound(Exception):
    """ sometimes a subterm index is larger than term length """


def subterm_by_index(atom: Proposition, index: int) -> Term:
    """
    extract a subterm using depth-first search through the tree of logical
    operations

    >>> atom = Predicate("this_is_a_test_case", [Function("f", [Variable("X")]), Function("g", [Variable("Y")])])
    >>> subterm_by_index(atom, 0)
    Traceback (most recent call last):
     ...
    ValueError: subterm with index 0 exists only for terms, but got: Predicate(name='this_is_a_test_case', arguments=[Function(name='f', arguments=[Variable(name='X')]), Function(name='g', arguments=[Variable(name='Y')])])
    >>> subterm_by_index(atom, 1) == atom.arguments[0]
    True
    >>> subterm_by_index(atom, 2) == atom.arguments[0].arguments[0]
    True
    >>> subterm_by_index(atom, 4) == atom.arguments[1].arguments[0]
    True

    :param atom: a predicate or a term
    :param index: an index of a desired subterm
    :returns: a subterm
    """
    if index == 0:
        if isinstance(atom, (Function, Variable)):
            return atom
        raise ValueError(
            f"subterm with index 0 exists only for terms, but got: {atom}"
        )
    subterm_length = 1
    if not isinstance(atom, Variable):
        for argument in atom.arguments:
            try:
                return pickle_copy(
                    subterm_by_index(argument, index - subterm_length)
                )
            except NoSubtermFound as error:
                subterm_length += error.args[0]
    raise NoSubtermFound(subterm_length)


class CantReplaceTheWholeTerm(Exception):
    """ an exception raised when trying to replace a subterm with index 0 """


class TermSelfReplace(Exception):
    """ an exception raised when trying to replace a subterm with itself """


def _replace_if_not_the_same(
    atom: Proposition, index: int, term: Term
) -> None:
    if not isinstance(atom, Variable):
        if atom.arguments[index] == term:
            raise TermSelfReplace
        atom.arguments[index] = term


def replace_subterm_by_index(
    atom: Proposition, index: int, term: Term
) -> None:
    """
    replace a subterm with a given index (depth-first search) by a new term
    replacement always happens inplace!

    >>> atom = Predicate("this_is_a_test_case", [Function("f", [Variable("X")]), Function("g", [Variable("Y")])])
    >>> replace_subterm_by_index(atom, 0, Variable("Z"))
    Traceback (most recent call last):
     ...
    gym_saturation.logic_ops.utils.NoSubtermFound: 5
    >>> replace_subterm_by_index(atom, 4, Function("h", [Variable("Z")]))
    >>> atom
    Predicate(name='this_is_a_test_case', arguments=[Function(name='f', arguments=[Variable(name='X')]), Function(name='g', arguments=[Function(name='h', arguments=[Variable(name='Z')])])])
    >>> replace_subterm_by_index(Predicate("this_is_a_test_case", [Variable("X")]), 1, Variable("X"))
    Traceback (most recent call last):
     ...
    gym_saturation.logic_ops.utils.TermSelfReplace

    :param atom: a predicate or a term
    :param index: an index of a subterm to replace, must be greater than 0
    :param term: replacement term for a given index
    :returns:
    """
    subterm_length = 1
    if not isinstance(atom, Variable):
        for i, argument in enumerate(atom.arguments):
            if index == subterm_length:
                _replace_if_not_the_same(atom, i, term)
                return
            try:
                replace_subterm_by_index(
                    argument, index - subterm_length, term
                )
                return
            except NoSubtermFound as error:
                subterm_length += error.args[0]
    raise NoSubtermFound(subterm_length)


def reduce_to_proof(clauses: List[Clause]) -> List[Clause]:
    """
    leave only clauses belonging to the refutational proof

    >>> reduce_to_proof([Clause([])])
    Traceback (most recent call last):
     ...
    ValueError: wrong refutational proof
    >>> state = [Clause([], label="one", processed=True)]
    >>> reduce_to_proof(state) == state
    True

    :param clauses: a list of clauses with labels and inference records
    :returns: the reduced list of clauses
    """
    state_dict = {clause.label: clause for clause in clauses}
    empty_clauses = [
        clause
        for label, clause in state_dict.items()
        if clause.literals == [] and clause.processed
    ]
    if len(empty_clauses) == 1:
        if empty_clauses[0].label is not None:
            reduced = []
            new_reduced = [empty_clauses[0]]
            while len(new_reduced) > 0:
                reduced += new_reduced
                new_reduced = [
                    state_dict[label]
                    for label in chain(
                        *[
                            (
                                []
                                if clause.inference_parents is None
                                else clause.inference_parents
                            )
                            for clause in new_reduced
                        ]
                    )
                ]
            return reduced
    raise ValueError("wrong refutational proof")
