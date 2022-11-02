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
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, Optional, Tuple
from uuid import uuid1

FALSEHOOD_SYMBOL = "$false"


@dataclass(frozen=True)
class Clause:
    """
    Clause is a disjunction of literals.

    .. _Clause:


    :param literals: a list of literals, forming the clause
    :param label: comes from the problem file or starts with ``inferred_`` if
         inferred during the episode
    :param role: formula role: axiom, hypothesis, ...
    :param inference_parents: a list of labels from which the clause was
         inferred. For clauses from the problem statement, this list is empty
    :param inference_rule: the rule according to which the clause was got from
         the ``inference_parents``
    :param processed: Boolean value splitting clauses into unprocessed and
         processed ones; in the beginning, everything is not processed
    :param birth_step: a number of the step when the clause appeared in the
         unprocessed set; clauses from the problem have ``birth_step`` zero
    """

    literals: str
    label: str = field(
        default_factory=lambda: "x" + str(uuid1()).replace("-", "_")
    )
    role: str = "lemma"
    inference_parents: Optional[Tuple[str, ...]] = None
    inference_rule: Optional[str] = None
    processed: Optional[bool] = None
    birth_step: Optional[int] = None


def pretty_print(clause: Clause) -> str:
    """
    Print a logical formula back to TPTP language.

    :param clause: a logical clause to print
    :returns: a TPTP string
    """
    res = f"cnf({clause.label}, {clause.role}, "
    res += clause.literals
    if (
        clause.inference_parents is not None
        and clause.inference_rule is not None
    ):
        res += (
            f", inference({clause.inference_rule}, [], ["
            + ", ".join(clause.inference_parents)
            + "])"
        )
    return res + ")."


class NoProofFoundError(Exception):
    """Exception raised when proof is requested but not found yet."""


def _flat_list(list_of_lists: Tuple[Tuple[Any, ...], ...]) -> Tuple[Any, ...]:
    return tuple(set(chain(*list_of_lists)))


def reduce_to_proof(
    clauses: Dict[str, Clause], goal: str = FALSEHOOD_SYMBOL
) -> Tuple[Clause, ...]:
    """
    Leave only clauses belonging to the refutation proof.

    >>> reduce_to_proof({
    ...     "one": Clause(FALSEHOOD_SYMBOL, label="one"),
    ...     "two": Clause(FALSEHOOD_SYMBOL, label="two")
    ... })
    Traceback (most recent call last):
     ...
    gym_saturation.utils.NoProofFoundError
    >>> state = {"one": Clause(FALSEHOOD_SYMBOL, label="one")}
    >>> reduce_to_proof(state) == (Clause(FALSEHOOD_SYMBOL, label="one"), )
    True

    :param clauses: a map of clause labels to clauses
    :param goal: literals of a goal clause (``$false`` by default)
    :returns: the reduced list of clauses
    :raises NoProofFoundError: if there is no complete refutation proof
        in a given proof state
    """
    empty_clauses = tuple(
        clause for clause in clauses.values() if clause.literals == goal
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
    raise NoProofFoundError


def get_tstp_proof(state: Dict[str, Clause]) -> str:
    """
    Return TSTP proof (if found; raises an error otherwise).

    :param state: map of clause labels to clauses
    """
    return "\n".join(
        reversed(
            [
                pretty_print(clause)
                for clause in reduce_to_proof(state)
                if clause.inference_rule is not None
            ]
        )
    )


def get_positive_actions(
    state: Dict[str, Clause], goal: str = FALSEHOOD_SYMBOL
) -> Tuple[int, ...]:
    """
    Return a sequence of actions which contributed to the proof found.

    If there is no proof yet, raises an error.

    :param state: map of clause labels to clauses
    :param goal: literals of the goal clause (``$false`` by default)
    """
    proof = reduce_to_proof(state, goal)
    return tuple(
        action
        for action, clause in enumerate(state.values())
        if clause in proof
    )
