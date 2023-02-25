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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gym_saturation.envs.saturation_env import (
    MAX_CLAUSES,
    STATE_DIFF_UPDATED,
    SaturationEnv,
)
from gym_saturation.utils import FALSEHOOD_SYMBOL
from gym_saturation.vampire_wrapper import VampireWrapper


class VampireEnv(SaturationEnv):
    """
    An RL environment around a Vampire prover.

    This class has the same API as SaturationEnv but uses another back-end.
    Here we have only a simple smoke test.

    >>> import sys
    >>> if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> vampire_binary = files("gym_saturation").joinpath(
    ...     os.path.join("resources", "vampire-mock")
    ... )
    >>> vampire_env = VampireEnv(["test"], vampire_binary_path=vampire_binary)
    >>> vampire_env.reset()
    Traceback (most recent call last):
     ...
    ValueError: ('Unexpected response type: ', 'who could expect that?')
    >>> from glob import glob
    >>> set_problems = sorted(glob(os.path.join(files("gym_saturation")
    ...     .joinpath(os.path.join("resources", "TPTP-mock", "Problems")),
    ... "SET", "*-*.p")))
    >>> vampire_env = VampireEnv(set_problems)
    >>> observation, info = vampire_env.reset()
    >>> for action in [0, 3, 6, 7, 8, 9, 10]:
    ...     observation, reward, terminated, truncated, info = (
    ...         vampire_env.step(action))
    >>> print(reward, terminated, truncated)
    1.0 True False

    test of a problem which is solver immediately after `reset`

    >>> problems = sorted(glob(os.path.join(files("gym_saturation").joinpath(
    ...     os.path.join("resources", "TPTP-mock", "Problems")
    ... ), "TST", "TST004-1.p")))
    >>> vampire_env = VampireEnv(problems)
    >>> observation, info = vampire_env.reset()
    >>> obs, reward, terminated, truncated, info = vampire_env.step(0)
    >>> print(reward, terminated, truncated)
    1.0 True False

    we can also run a full Gymnasium environment check

    >>> from gymnasium.utils.env_checker import check_env
    >>> import gymnasium as gym
    >>> env = gym.make(
    ...     "Vampire-v0",
    ...     problem_list=set_problems,
    ...     max_clauses=9
    ... )
    >>> check_env(env.unwrapped)
    cnf(1, ...).
    ...
    cnf(9, ...).
    """

    def __init__(
        self,
        problem_list: List[str],
        max_clauses: int = MAX_CLAUSES,
        render_mode: str = "human",
        vampire_binary_path: str = "vampire",
    ):
        """
        Initialise a :ref:`VampireWrapper <vampire-wrapper>`.

        :param problem_list: a list of names of TPTP problem files
        :param max_clauses: maximal number of clauses in proof state
        :param vampire_binary_path: a path to Vampire binary;
            by default we expect it to be in the $PATH
        """
        super().__init__(problem_list, max_clauses, render_mode)
        self._vampire = VampireWrapper(vampire_binary_path)
        self._step = 0

    def _update_processed(
        self, clause_label: str, updated: Dict[str, Dict[str, Any]]
    ) -> None:
        if clause_label in updated:
            updated[clause_label].update({"processed": 1})
        else:
            self.state[clause_label].update({"processed": 1})
            updated.update({clause_label: self.state[clause_label]})

    def _parse_vampire_response(
        self, vampire_response: Tuple[Tuple[str, str, str], ...]
    ) -> Dict[str, Dict[str, Any]]:
        updated: Dict[str, Dict[str, Any]] = {}
        for response_type, clause_label, clause_text in vampire_response:
            if response_type in {"new", "final", "input", "fn def discovered"}:
                updated[clause_label] = self._parse_vampire_clause(
                    clause_label, clause_text
                )
            elif response_type in {
                "active",
                "forward reduce",
                "backward reduce",
            }:
                self._update_processed(clause_label, updated)
            elif response_type not in {"passive", "new propositional"}:
                raise ValueError("Unexpected response type: ", response_type)
        return updated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Any]]:  # noqa: D102
        super().reset(seed=seed)
        tptp_folder = os.path.join(
            os.path.dirname(self.problem_filename), "..", ".."
        )
        vampire_response = self._vampire.start(
            self.problem_filename, tptp_folder
        )
        self.state = {}
        self._step = 0
        updated = self._parse_vampire_response(vampire_response)
        self.state = updated
        return tuple(self.state.values()), {STATE_DIFF_UPDATED: self.state}

    def _do_deductions(self, action: np.int64) -> Dict[str, Dict[str, Any]]:
        if any(
            clause["literals"] == FALSEHOOD_SYMBOL
            for clause in self.state.values()
        ):
            return {}
        given_clause = list(self.state.values())[action]
        updated = self._parse_vampire_response(
            self._vampire.pick_a_clause(given_clause["label"])
        )
        self._step += 1
        self.state.update(updated)
        return updated

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
            "processed": 0,
            "birth_step": self._step,
        }
