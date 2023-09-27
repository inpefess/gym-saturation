#   Copyright 2023 Boris Shminke
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# noqa: D205, D400
"""
MultiDiscrete-to-Discrete Wrapper
==================================
"""
import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper


class Md2DWrapper(ActionWrapper):
    """
    A wrapper transforming a MultiDiscrete action space to a Discrete one.

    >>> env = gym.make("Vampair-v0", max_clauses=10)
    >>> wrapped_env = Md2DWrapper(env)
    >>> wrapped_env.action_space
    Discrete(100)
    >>> _ = wrapped_env.reset()
    >>> _ = wrapped_env.step(2)  # (0, 2)
    >>> _, reward, _, _, _ = wrapped_env.step(35)  # (3, 5)
    >>> reward
    1.0
    """

    def __init__(self, env: gym.Env):
        """Redefine the action space."""
        super().__init__(env)
        self.nvec = self.env.action_space.nvec  # type: ignore
        self.action_space = gym.spaces.Discrete(self.nvec.prod())

    def action(self, action: np.int64) -> np.ndarray:
        """Return a modified action before ``step`` is called.

        :param action: The original actions
        :returns: The modified actions
        """
        remainder = int(action)
        new_action = np.zeros_like(self.nvec)
        for i in range(new_action.shape[0]):
            new_action[i] = remainder % int(self.nvec[i])
            remainder //= int(self.nvec[i])
        return new_action
