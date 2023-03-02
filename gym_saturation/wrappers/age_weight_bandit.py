#   Copyright 2023 Boris Shminke
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# noqa: D205, D400
"""
Age-Weight Bandit
=================
"""
import gymnasium as gym
import numpy as np

from gym_saturation.envs.saturation_env import SaturationEnv


class AgeWeightBandit(gym.ActionWrapper):
    """
    A wrapper for a saturation prover with only two actions.

    .. _age_weight_bandit:

    ``0`` --- select the oldest clause
    ``1`` --- select the shortest clause

    >>> import os
    >>> tptp_folder = getfixture("mock_tptp_folder")  # noqa: F821
    >>> problem_list = [
    ...     os.path.join(tptp_folder, "Problems", "TST", "TST003-1.p")
    ... ]
    >>> import gymnasium as gym
    >>> env = gym.make("Vampire-v0", problem_list=problem_list, max_clauses=9)
    >>> bandit_env = AgeWeightBandit(env)
    >>> _ = bandit_env.reset()
    >>> observation, _, _, _, _ = bandit_env.step(0)
    >>> observation["action_mask"]
    array([0., 1., 1., 0., 0., 0., 0., 0., 0.], dtype=float32)
    >>> observation, _, _, _, _ = bandit_env.step(1)
    >>> observation["action_mask"]
    array([0., 1., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)
    >>> _ = bandit_env.step(2)
    Traceback (most recent call last):
    ...
    ValueError: Impossible action: 2
    """

    action_space = gym.spaces.Discrete(2)

    def action(self, action: int) -> np.int64:  # type: ignore
        """
        Modify action before ``step`` is called.

        :param action: The original ``step`` action
        :returns: The modified action
        :raises ValueError: if an ``action`` is anything except ``0`` or ``1``
        """
        env: SaturationEnv = self.env  # type: ignore
        if action == 0:
            return env.state.action_mask.argmax()
        if action == 1:
            clauses = env.state.clauses[: env.state.action_mask.shape[0]]
            return (
                np.pad(
                    np.array(
                        [1 / len(clause["literals"]) for clause in clauses]
                    ),
                    (0, env.state.action_mask.shape[0] - len(clauses)),
                )
                * env.state.action_mask
            ).argmax()
        raise ValueError(f"Impossible action: {action}")
