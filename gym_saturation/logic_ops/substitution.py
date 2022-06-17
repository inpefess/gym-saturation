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
Substitution
=============
"""
import dataclasses
from typing import Union

from tptp_lark_parser import grammar


@dataclasses.dataclass
class Substitution:
    """
    A mapping from ``Variable`` to ``Term``.

    >>> substitution = Substitution(
    ...     grammar.Variable(0), grammar.Function(2, ())
    ... )
    >>> substitution(
    ...     grammar.Clause((
    ...         grammar.Literal(
    ...             False,
    ...             grammar.Predicate(
    ...                 3,
    ...                 (grammar.Function(1, (grammar.Variable(0),)),))
    ...             ),
    ...     ), label="this_is_a_test_case")
    ... )
    Clause(literals=(Literal(negated=False, atom=Predicate(index=3, arguments=(Function(index=1, arguments=(Function(index=2, arguments=()),)),))),), label='this_is_a_test_case', role='lemma', inference_parents=None, inference_rule=None, processed=None, birth_step=None)
    >>> substitution(grammar.Variable(0))
    Function(index=2, arguments=())
    """

    variable: grammar.Variable
    term: grammar.Term

    def __call__(
        self, target: Union[grammar.Clause, grammar.Proposition]
    ) -> Union[grammar.Clause, grammar.Proposition]:
        """
        Apply a substitution to a given target clause.

        :param target: a clause to which to apply the substitution
        """
        if isinstance(target, grammar.Clause):
            return self.substitute_in_clause(target)
        if isinstance(target, grammar.Predicate):
            return self._substitute_in_predicate(target)
        return self._substitute_in_term(target)

    def _substitute_in_term(self, term: grammar.Term) -> grammar.Term:
        if isinstance(term, grammar.Function):
            return grammar.Function(
                term.index,
                tuple(
                    self._substitute_in_term(argument)
                    for argument in term.arguments
                ),
            )
        if term.index == self.variable.index:
            return self.term
        return term

    def _substitute_in_predicate(
        self, predicate: grammar.Predicate
    ) -> grammar.Predicate:
        return grammar.Predicate(
            predicate.index,
            tuple(
                self._substitute_in_term(argument)
                for argument in predicate.arguments
            ),
        )

    def substitute_in_clause(self, clause: grammar.Clause) -> grammar.Clause:
        """
        Apply the substitution to something which is known to be a ``Clause``.

        :param clause: a clause to apply substitution to
        :returns: the result of the substitution
        """
        literals = tuple(
            grammar.Literal(
                literal.negated,
                self._substitute_in_predicate(literal.atom),
            )
            for literal in clause.literals
        )
        return dataclasses.replace(clause, literals=literals)
