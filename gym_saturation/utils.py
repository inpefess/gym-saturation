# Copyright 2021-2023 Boris Shminke
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
from itertools import chain
from typing import Any, Dict, Tuple

FALSEHOOD_SYMBOL = "$false"


def pretty_print(clause: Dict[str, Any]) -> str:
    """
    Print a logical formula back to TPTP language.

    :param clause: a logical clause to print
    :returns: a TPTP string
    """
    res = f"cnf({clause['label']}, {clause['role']}, "
    res += clause["literals"]
    res += (
        f", inference({clause['inference_rule']}, [], ["
        + ", ".join(clause["inference_parents"])
        + "])"
    )
    return res + ")."


class NoProofFoundError(Exception):
    """Exception raised when proof is requested but not found yet."""


def _flat_list(list_of_lists: Tuple[Tuple[Any, ...], ...]) -> Tuple[Any, ...]:
    return tuple(set(chain(*list_of_lists)))


def reduce_to_proof(
    clauses: Tuple[Dict[str, Any], ...], goal: str = FALSEHOOD_SYMBOL
) -> Tuple[Dict[str, Any], ...]:
    """
    Leave only clauses belonging to the refutation proof.

    >>> one = {
    ...     "literals": FALSEHOOD_SYMBOL, "label": "one",
    ...     "inference_parents": ()
    ... }
    >>> reduce_to_proof((one, {"literals": FALSEHOOD_SYMBOL, "label": "two"}))
    Traceback (most recent call last):
     ...
    gym_saturation.utils.NoProofFoundError
    >>> state = (one, )
    >>> reduce_to_proof(state) == state
    True

    :param clauses: a map of clause labels to clauses
    :param goal: literals of a goal clause (``$false`` by default)
    :returns: the reduced list of clauses
    :raises NoProofFoundError: if there is no complete refutation proof
        in a given proof state
    """
    empty_clauses = tuple(
        clause for clause in clauses if clause["literals"] == goal
    )
    if len(empty_clauses) == 1:
        reduced: Tuple[Dict[str, Any], ...] = ()
        new_reduced: Tuple[Dict[str, Any], ...] = (empty_clauses[0],)
        while len(new_reduced) > 0:
            reduced += tuple(
                clause for clause in new_reduced if clause not in reduced
            )
            new_reduced = tuple(
                clause
                for clause in clauses
                if clause["label"]
                in tuple(
                    reversed(
                        sorted(
                            _flat_list(
                                tuple(
                                    clause["inference_parents"]
                                    for clause in new_reduced
                                )
                            )
                        )
                    )
                )
            )
        return reduced
    raise NoProofFoundError


def get_tstp_proof(state: Tuple[Dict[str, Any], ...]) -> str:
    """
    Return TSTP proof (if found; raises an error otherwise).

    :param state: tuple of clauses
    :return: a TSTP formatted string
    """
    return "\n".join(
        reversed([pretty_print(clause) for clause in reduce_to_proof(state)])
    )


def get_positive_actions(
    state: Tuple[Dict[str, Any], ...], goal: str = FALSEHOOD_SYMBOL
) -> Tuple[int, ...]:
    """
    Return a sequence of actions which contributed to the proof found.

    If there is no proof yet, raises an error.

    :param state: map of clause labels to clauses
    :param goal: literals of the goal clause (``$false`` by default)
    """
    proof = reduce_to_proof(state, goal)
    return tuple(
        action for action, clause in enumerate(state) if clause in proof
    )
