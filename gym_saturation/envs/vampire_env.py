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
Saturation Environment with Vampire back-end
============================================
"""
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gym_saturation.envs.saturation_env import (
    ACTION_MASK,
    MAX_CLAUSES,
    REAL_OBS,
    SaturationEnv,
)
from gym_saturation.utils import FALSEHOOD_SYMBOL, MOCK_TPTP_FOLDER
from gym_saturation.vampire_wrapper import VampireWrapper


class VampireEnv(SaturationEnv):
    """
    An RL environment around a Vampire prover.

    This class has the same API as SaturationEnv but uses another back-end.
    Here we have only a simple smoke test.

    >>> from gym_saturation.utils import MOCK_TPTP_FOLDER
    >>> vampire_binary = os.path.join(MOCK_TPTP_FOLDER, "..", "vampire-mock")
    >>> vampire_env = VampireEnv(vampire_binary_path=vampire_binary)
    >>> vampire_env.reset()
    Traceback (most recent call last):
     ...
    ValueError: ('Unexpected response type: ', 'who could expect that?')
    >>> from glob import glob
    >>> set_problems = sorted(glob(os.path.join(MOCK_TPTP_FOLDER, "Problems",
    ...     "SET", "*-*.p")))
    >>> vampire_env = VampireEnv()
    >>> vampire_env.set_task(set_problems[0])
    >>> observation, info = vampire_env.reset()
    >>> for action in [0, 3, 6, 7, 8, 9, 10]:
    ...     observation, reward, terminated, truncated, info = (
    ...         vampire_env.step(action))
    >>> print(reward, terminated, truncated)
    1.0 True False

    test of a problem which is solver immediately after `reset`

    >>> problem_filename = os.path.join(
    ...     MOCK_TPTP_FOLDER, "Problems", "TST", "TST004-1.p")
    >>> vampire_env = VampireEnv()
    >>> vampire_env.set_task(problem_filename)
    >>> observation, info = vampire_env.reset()
    >>> obs, reward, terminated, truncated, info = vampire_env.step(0)
    >>> print(int(reward), terminated, truncated)
    1 True False

    we can also run a full Gymnasium environment check

    >>> from gymnasium.utils.env_checker import check_env
    >>> import gymnasium as gym
    >>> env = gym.make(
    ...     "Vampire-v0",
    ...     max_clauses=9
    ... ).unwrapped
    >>> env.set_task(set_problems[0])
    >>> check_env(env)
    cnf(1, ...).
    ...
    cnf(9, ...).
    """

    def __init__(
        self,
        max_clauses: int = MAX_CLAUSES,
        render_mode: str = "human",
        vampire_binary_path: str = "vampire",
    ):
        """
        Initialise a :ref:`VampireWrapper <vampire-wrapper>`.

        :param max_clauses: maximal number of clauses in proof state
        :param vampire_binary_path: a path to Vampire binary;
            by default we expect it to be in the $PATH
        """
        super().__init__(max_clauses, render_mode)
        self._vampire = VampireWrapper(vampire_binary_path)
        self._task = os.path.join(
            MOCK_TPTP_FOLDER, "Problems", "TST", "TST003-1.p"
        )

    def _parse_vampire_response(
        self, vampire_response: Tuple[Tuple[str, str, str], ...]
    ) -> None:
        for response_type, clause_label, clause_text in vampire_response:
            if response_type in {"new", "final", "input", "fn def discovered"}:
                self.state.add_clause(
                    self._parse_vampire_clause(clause_label, clause_text)
                )
            elif response_type in {
                "active",
                "forward reduce",
                "backward reduce",
            }:
                self.state.set_action_mask_by_label(clause_label, 0.0)
            elif response_type == "passive":
                self.state.set_action_mask_by_label(clause_label, 1.0)
            elif response_type != "new propositional":
                raise ValueError("Unexpected response type: ", response_type)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # noqa: D102
        super().reset(seed=seed)
        tptp_folder = os.path.join(
            os.path.dirname(self.get_task()), "..", ".."
        )
        vampire_response = self._vampire.start(self.get_task(), tptp_folder)
        self._parse_vampire_response(vampire_response)
        return {
            REAL_OBS: tuple(self.state.clauses),
            ACTION_MASK: self.state.action_mask,
        }, {}

    def _do_deductions(self, action: np.int64) -> None:
        if any(
            clause["literals"] == FALSEHOOD_SYMBOL
            for clause in self.state.clauses
        ):
            return
        self._parse_vampire_response(
            self._vampire.pick_a_clause(self.state.clause_labels[action])
        )

    def _parse_vampire_clause(
        self, clause_label: str, clause_text: str
    ) -> Dict[str, Any]:
        formula, inference_info = clause_text.split("[")
        pre_inference = inference_info.split("]")[0].split(" ")
        if len(pre_inference) > 1:
            inference_parents = tuple(pre_inference[-1].split(","))
            inference_rule = "_".join(pre_inference[:-1])
        else:
            inference_parents, inference_rule = (), pre_inference[0]
        return {
            "literals": formula.strip(),
            "label": clause_label,
            "role": "lemma",
            "inference_rule": inference_rule,
            "inference_parents": inference_parents,
            "birth_step": self.state.step_number,
        }
