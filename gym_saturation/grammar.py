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
# type: ignore
"""
Grammar
********
"""
from typing import List, NamedTuple, Optional, Union
from uuid import uuid1


class Variable(NamedTuple):
    """
    .. _variable:

    a variable is characterised only by its name
    """

    name: str

    def todict(self) -> dict:
        """
        :returns: a value similar to `_asdict` but with fields represented
            recursively as dicts too
        """
        return self._asdict()  # pylint: disable=no-member


class Function(NamedTuple):
    """
    .. _Function:

    a functional symbol might be applied to a list of arguments
    """

    name: str
    arguments: List[Union[Variable, "Function"]]

    def todict(self) -> dict:
        """
        :returns: a value similar to `_asdict` but with fields represented
            recursively as dicts too
        """
        return {
            "name": self.name,
            "arguments": [argument.todict() for argument in self.arguments],
        }


Term = Union[Variable, Function]
Term.__doc__ = """
.. _Term:

Term is either a :ref:`Variable <Variable>` or a :ref:`Function <Function>`
"""


class Predicate(NamedTuple):
    """
    .. _Predicate:

    a predicate symbol might be applied to a list of arguments
    """

    name: str
    arguments: List[Term]

    def todict(self) -> dict:
        """
        :returns: a value similar to `_asdict` but with fields represented
            recursively as dicts too
        """
        return {
            "name": self.name,
            "arguments": [argument.todict() for argument in self.arguments],
        }


Proposition = Union[Predicate, Term]
Proposition.__doc__ = """
.. _Proposition:

Proposition is either a :ref:`Predicate <Predicate>` or a :ref:`Term <Term>`
"""


class Literal(NamedTuple):
    """
    .. _Literal:

    literal is an atom which can be negated or not
    """

    negated: bool
    atom: Predicate

    def todict(self) -> dict:
        """
        :returns: a value similar to `_asdict` but with fields represented
            recursively as dicts too
        """
        return {"negated": self.negated, "atom": self.atom.todict()}


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


class Clause(NamedTuple):
    """
    .. _Clause:

    clause is a disjunction of literals

    :param literals: a list of literals, forming the clause
    :param label: comes from the problem file or starts with ``inferred_`` if
         inferred during the episode
    :param role: formula role (axiom, hypothesis, etc)
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
    label: Optional[str] = None
    role: str = "lemma"
    inference_parents: Optional[List[str]] = None
    inference_rule: Optional[str] = None
    processed: Optional[bool] = None
    birth_step: Optional[int] = None

    def __repr__(self):
        res = f"cnf({self.label}, {self.role}, "
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

    def todict(self) -> dict:
        """
        :returns: a value similar to `_asdict` but with fields represented
            recursively as dicts too
        """
        return {
            "literals": [literal.todict() for literal in self.literals],
            "label": self.label,
            "role": self.role,
            "birth_step": self.birth_step,
            "inference_parents": self.inference_parents,
            "inference_rule": self.inference_rule,
            "processed": self.processed,
        }


def new_clause(
    literals: List[Literal], label: Optional[str] = None, **kwargs
) -> Clause:
    """
    a trivial clause factory

    :param literals: a list of literals
    :param label: if empty, it will be randomly generated
    :returns: a new clause
    """
    if label is None:
        new_label = "x" + str(uuid1()).replace("-", "_")
    else:
        new_label = label
    return Clause(literals=literals, label=new_label, **kwargs)
