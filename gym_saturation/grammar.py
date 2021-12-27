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
from dataclasses import dataclass, field
from typing import List, Optional, Union
from uuid import uuid1


@dataclass
class Variable:
    """ a variable is characterised only by its name """

    name: str


@dataclass
class Function:
    """ a functional symbol might be applied to a list of arguments """

    name: str
    arguments: List[Union[Variable, "Function"]]


Term = Union[Variable, Function]


@dataclass
class Predicate:
    """ a predicate symbol might be applied to a list of arguments """

    name: str
    arguments: List[Term]


Proposition = Union[Predicate, Term]


@dataclass
class Literal:
    """ literal is an atom which can be negated or not """

    negated: bool
    atom: Predicate


def _term_to_tptp(term: Term) -> str:
    if isinstance(term, Function):
        arguments = [_term_to_tptp(argument) for argument in term.arguments]
        if arguments != []:
            return f"{term.name}({','.join(arguments)})"
    return term.name


def _literal_to_tptp(literal: Literal) -> str:
    res = "~" if literal.negated else ""
    if literal.atom.name != "=":
        res += (
            literal.atom.name
            + "("
            + ", ".join(
                [_term_to_tptp(term) for term in literal.atom.arguments]
            )
            + ")"
        )
    else:
        res += (
            _term_to_tptp(literal.atom.arguments[0])
            + " = "
            + _term_to_tptp(literal.atom.arguments[1])
        )
    return res


@dataclass
class Clause:
    """
    clause is a disjunction of literals

    :param literals: a list of literals, forming the clause
    :param label: comes from the problem file or starts with ``inferred_`` if
         inferred during the episode
    :param inference_parents: a list of labels from which the clause was
         inferred. For clauses from the problem statement, this list is empty
    :param inference_rule: the rule according to which the clause was got from
         the ``inference_parents``
    :param processed: boolean value splitting clauses into unprocessed and
         processed ones; in the beginning, everything is not processed
    :param birth_step: a number of the step when the clause appeared in the
         unprocessed set; clauses from the problem have ``birth_step`` zero
    """

    literals: List[Literal]
    label: str = field(default_factory=lambda: str(uuid1()))
    inference_parents: Optional[List[str]] = None
    inference_rule: Optional[str] = None
    processed: Optional[bool] = None
    birth_step: Optional[int] = None

    def __repr__(self):
        res = f"cnf({self.label}, hypothesis, "
        for literal in self.literals:
            res += _literal_to_tptp(literal) + " | "
        if res[-2:] == "| ":
            res = res[:-3]
        if not self.literals:
            res += "$false"
        if (
            self.inference_parents is not None
            and self.inference_rule is not None
        ):
            res += (
                f", inference({self.inference_rule}, [], ["
                + ", ".join(self.inference_parents)
                + "])"
            )
        return res + ")."
