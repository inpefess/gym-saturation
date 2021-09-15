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
from dataclasses import dataclass
from typing import Union

from gym_saturation import grammar
from gym_saturation.utils import pickle_copy


@dataclass
class Substitution:
    """
    a mapping from ``Variable`` to ``Term``

    >>> substitution = Substitution(grammar.Variable("X"), grammar.Function("this_is_a_test_case", []))
    >>> substitution(grammar.Clause([grammar.Literal(False, grammar.Predicate("p", [grammar.Function("this_is_a_test_case", [grammar.Variable("X")])]))]))
    Clause(literals=[Literal(negated=False, atom=Predicate(name='p', arguments=[Function(name='this_is_a_test_case', arguments=[Function(name='this_is_a_test_case', arguments=[])])]))], label=None, inference_parents=None, inference_rule=None, processed=None, birth_step=None)
    >>> substitution(grammar.Variable("X"))
    Function(name='this_is_a_test_case', arguments=[])
    """

    variable: grammar.Variable
    term: grammar.Term

    def __call__(
        self, target: Union[grammar.Clause, grammar.Proposition]
    ) -> Union[grammar.Clause, grammar.Proposition]:
        if isinstance(target, grammar.Clause):
            return self.substitute_in_clause(target)
        if isinstance(target, grammar.Predicate):
            return self._substitute_in_predicate(target)
        return self._substitute_in_term(target)

    def _substitute_in_term(self, term: grammar.Term) -> grammar.Term:
        if isinstance(term, grammar.Function):
            return grammar.Function(
                term.name,
                [
                    self._substitute_in_term(argument)
                    for argument in term.arguments
                ],
            )
        if term.name == self.variable.name:
            return pickle_copy(self.term)
        return pickle_copy(term)

    def _substitute_in_predicate(
        self, predicate: grammar.Predicate
    ) -> grammar.Predicate:
        return grammar.Predicate(
            predicate.name,
            [
                self._substitute_in_term(argument)
                for argument in predicate.arguments
            ],
        )

    def substitute_in_clause(self, clause: grammar.Clause) -> grammar.Clause:
        """
        apply the substitution to something which is known to be a ``Clause``

        :param clause: a clause to apply substitution to
        :returns: the result of the substitution
        """
        literals = []
        for literal in clause.literals:
            literals.append(
                grammar.Literal(
                    literal.negated,
                    self._substitute_in_predicate(literal.atom),
                )
            )
        new_clause: grammar.Clause = pickle_copy(clause)
        new_clause.literals = literals
        return new_clause
