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

BARBARA = (
    {
        "label": "all_men_are_mortal",
        "role": "hypothesis",
        "literals": "~man(X) | mortal(X)",
        "inference_rule": "input",
        "inference_parents": [],
        "birth_step": 0,
    },
    {
        "label": "socrates_is_a_man",
        "role": "hypothesis",
        "literals": "man(socrates)",
        "inference_rule": "input",
        "inference_parents": [],
        "birth_step": 0,
    },
    {
        "label": "socrates_is_mortal",
        "role": "negated_conjecture",
        "literals": "mortal(socrates)",
        "inference_rule": "input",
        "inference_parents": [],
        "birth_step": 0,
    },
)


class DummySaturationEnv(SaturationEnv):
    """Dummy implementation for tests."""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # noqa: D102
        super().reset(seed=seed)
        self.state = ProofState(
            clauses=list(BARBARA),
            clause_labels=[str(clause["label"]) for clause in BARBARA],
            action_mask=np.pad(
                np.ones((len(BARBARA),), dtype=np.float32),
                pad_width=(0, int(self.action_space.n) - len(BARBARA)),
            ),
            step_number=-1,
        )
        return {
            REAL_OBS: self.state.clauses,
            ACTION_MASK: self.state.action_mask,
        }, {}

    def _do_deductions(self, action: np.int64) -> None:
        if action == 1:
            self.state.clauses.append(
                {
                    "literals": "dummy",
                    "role": "lemma",
                    "label": "dummy",
                    "inference_rule": "dummy",
                    "inference_parents": [],
                }
            )
            self.state.clause_labels.append("dummy")
        if action == 2:
            self.state.clauses.append(
                {
                    "literals": FALSEHOOD_SYMBOL,
                    "role": "lemma",
                    "label": "falsehood",
                    "inference_rule": "dummy",
                    "inference_parents": [],
                }
            )
            self.state.clause_labels.append("falsehood")
            self.state.action_mask[len(self.state.clauses)] = 1.0
