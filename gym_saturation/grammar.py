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
from typing import List, Optional, Union


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
    arguments: List[Union[Function, Variable]]


Proposition = Union[Predicate, Term]


@dataclass
class Literal:
    """ literal is an atom which can be negated or not """

    negated: bool
    atom: Predicate


@dataclass
class Clause:
    """ clause is a disjunction of literals """

    literals: List[Literal]
    label: Optional[str] = None
    inference_parents: Optional[List[str]] = None
    inference_rule: Optional[str] = None
    processed: Optional[bool] = None
    birth_step: Optional[int] = None
