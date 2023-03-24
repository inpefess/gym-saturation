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
A dummy SaturationEnv implementation for tests
===============================================
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gym_saturation.envs.saturation_env import (
    ACTION_MASK,
    REAL_OBS,
    SaturationEnv,
)
from gym_saturation.proof_state import ProofState
from gym_saturation.utils import FALSEHOOD_SYMBOL

BASE_CLAUSE = {
    "label": "dummy",
    "role": "lemma",
    "inference_rule": "dummy",
    "inference_parents": (),
    "birth_step": 0,
}


class DummySaturationEnv(SaturationEnv):
    """Dummy implementation for tests."""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # noqa: D102
        super().reset(seed=seed)
        one = {**BASE_CLAUSE, **{"literals": "p(X)", "label": "one"}}
        two = {**BASE_CLAUSE, **{"literals": "p(Y)", "label": "two"}}
        three = {**BASE_CLAUSE, **{"literals": "p(Z)", "label": "three"}}
        four = {**BASE_CLAUSE, **{"literals": "~p(X)", "label": "four"}}
        self.state = ProofState(
            clauses=[one, two, three, four],
            clause_labels=["one", "two", "three", "four"],
            action_mask=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
            step_number=0,
        )
        return {
            REAL_OBS: self.state.clauses,
            ACTION_MASK: self.state.action_mask,
        }, {}

    def _do_deductions(self, action: np.int64) -> None:
        if action == 3:
            self.state.clauses.append(
                {
                    "literals": FALSEHOOD_SYMBOL,
                    "role": "lemma",
                    "label": "falsehood",
                    "inference_rule": "dummy",
                    "inference_parents": ("four",),
                }
            )
            self.state.clause_labels.append("falsehood")
            self.state.action_mask[4] = 1.0
