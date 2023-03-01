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

import numpy as np


@dataclass
class ProofState:
    """
    An object to store all relevant info about a saturation prover state.

    :param clauses: clauses (both processed and not) for access by index
    :param clause_labels: string labels (another way to address clauses)
    :param action_mask: a ``numpy`` array to separated processed from not
    :param step_number: current step number. ``-1`` before reset, ``0`` after
    """

    clauses: List[Dict[str, Any]]
    clause_labels: List[str]
    action_mask: np.ndarray
    step_number: int

    def add_clause(self, clause: Dict[str, Any]) -> None:
        """
        Add clause and its label to the state.

        :param clause: a clause to add (together with its label)
        """
        if clause["label"] not in self.clause_labels:
            self.clauses.append(clause)
            self.clause_labels.append(clause["label"])

    def set_action_mask_by_label(
        self, clause_label: str, mask_value: float
    ) -> None:
        """
        Set action mask value for a clause with a given label.

        If we get a label which belong to the clause added by breaking the
        maximal clauses number constraint, nothing happens.

        :param clause_label: a label of the clause to set action mask
        :param mask_value: ``0.0`` (processed/removed) or ``1.0`` (unprocessed)
        """
        clause_index = self.clause_labels.index(clause_label)
        if clause_index < self.action_mask.shape[0]:
            self.action_mask[clause_index] = mask_value
