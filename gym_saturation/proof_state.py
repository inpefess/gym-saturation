# Copyright 2023 Boris Shminke
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
Proof State
============
"""
from dataclasses import dataclass
from typing import Any, Dict, List

from gym_saturation.constants import FALSEHOOD_SYMBOL


@dataclass
class ProofState:
    """
    An object to store all relevant info about a saturation prover state.

    :param clauses: clauses (both processed and not) for access by index
    :param clause_labels: string labels (another way to address clauses)
    :param step_number: current step number. ``-1`` before reset, ``0`` after
    :param max_clauses: maximal possible number of clauses in the proof state
    """

    clauses: List[Dict[str, Any]]
    clause_labels: List[str]
    step_number: int
    max_clauses: int

    def add_clause(self, clause: Dict[str, Any]) -> None:
        """
        Add clause and its label to the state.

        :param clause: a clause to add (together with its label)
        """
        if clause["label"] not in self.clause_labels:
            self.clauses.append(clause)
            self.clause_labels.append(clause["label"])

    @property
    def terminated(self) -> bool:
        """Refutation found or satisfiability established."""
        return (
            max(
                clause["literals"] == FALSEHOOD_SYMBOL
                for clause in self.clauses
            )
        ) and not self.truncated

    @property
    def truncated(self) -> bool:
        """More clauses generated than expected."""
        return len(self.clauses) > self.max_clauses
