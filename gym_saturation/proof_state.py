# Copyright 2023-2025 Boris Shminke
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
from typing import Any

from gym_saturation.constants import FALSEHOOD_SYMBOL


@dataclass
class ProofState:
    """
    An object to store all relevant info about a saturation prover state.

    :param clauses: passive clauses
    """

    clauses: dict[str, dict[str, Any]]

    @property
    def terminated(self) -> bool:
        """Refutation found or satisfiability established."""
        return max(
            clause["literals"] == FALSEHOOD_SYMBOL
            for clause in self.clauses.values()
        )
