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
from typing import Any, List, Union

from gym_saturation.grammar import (
    Clause,
    Function,
    Predicate,
    Proposition,
    Term,
    Variable,
)
from gym_saturation.logic_ops.substitution import Substitution


def deduplicate(a_list: List[Any]) -> List[Any]:
    """
    deduplicate a list

    :param a_list: a list of possibly repeating items
    :returns: a list of unique items
    """
    new_list = list()
    for item in a_list:
        if item not in new_list:
            new_list.append(item)
    return new_list


def is_subterm(one: Proposition, two: Proposition) -> bool:
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
            if is_subterm(one, argument):
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
    variable_list = list()
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
    new_clauses = list()
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
            length += term_length(term)
        length += 1
    return length


def term_length(term: Term) -> int:
    """
    total length of subterms plus ones for a function or one for a variable

    :param term: a function or a variable
    :return: sctructural length of a term

    >>> term_length(Function("f", [Variable("X")]))
    2
    """
    length = 0
    if isinstance(term, Variable):
        return 1
    for subterm in term.arguments:
        length += term_length(subterm)
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
