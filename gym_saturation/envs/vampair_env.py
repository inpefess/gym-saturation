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
Environment with experimental pair-selection Vampire back-end
==============================================================
"""
import numpy as np
from gymnasium.spaces import MultiDiscrete

from gym_saturation.envs.saturation_env import MAX_CLAUSES
from gym_saturation.envs.vampire_env import VampireEnv
from gym_saturation.vampire_wrapper import VampireWrapper


class VampairEnv(VampireEnv):
    """
    An RL environment wrapper around an experimental Vampire prover.

    >>> from gymnasium.utils.env_checker import check_env
    >>> import gymnasium as gym
    >>> env = gym.make("Vampair-v0", max_clauses=5).unwrapped
    >>> check_env(env)
    cnf(1, ...).
    ...
    cnf(5, ...).
    """

    def __init__(
        self,
        max_clauses: int = MAX_CLAUSES,
        render_mode: str = "human",
        prover_binary_path: str = "vampair",
    ):
        """
        Initialise a :ref:`VampireWrapper <vampire-wrapper>`.

        :param max_clauses: maximal number of clauses in proof state
        :param render_mode: a mode of running ``render`` method
        :param prover_binary_path: a path to Vampire binary;
            by default we expect it to be in the $PATH
        """
        super().__init__(max_clauses, render_mode)
        self._vampire = VampireWrapper(
            prover_binary_path,
            command_line_arguments=" -sa given_pair --show_passive on"
            " --show_new on --time_limit 0 --avatar off ",
        )
        self.action_space = MultiDiscrete(
            [self.state.max_clauses, self.state.max_clauses]
        )

    def _do_deductions(self, action: np.ndarray) -> None:  # type: ignore
        clause_labels = list(self.state.clauses.keys())
        total_clauses = len(clause_labels)
        if action[0] < total_clauses and action[1] < total_clauses:
            self._parse_vampire_response(
                self._vampire.pick_a_clause(
                    f"{clause_labels[action[0]]} {clause_labels[action[1]]}"
                )
            )
