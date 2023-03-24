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
Constant Parametric Actions Wrapper
====================================
"""
from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType


class ConstantParametricActionsWrapper(gym.ObservationWrapper):
    """
    A wrapper which make the observation to contain constant action embeddings.

    .. _constant_parametric_actions:

    >>> env = ConstantParametricActionsWrapper(gym.make("CartPole-v1"))
    >>> observation, info = env.reset()
    >>> observation
    {'avail_actions': array([[1., 0.],
           [0., 1.]])}
    """

    def __init__(self, env: gym.Env, avail_actions_key: str = "avail_actions"):
        """Initialise the observation wrapper."""
        super().__init__(env)
        self.avail_actions_key = avail_actions_key
        self.env.observation_space = gym.spaces.Dict(
            {
                self.avail_actions_key: gym.spaces.Box(
                    0,
                    1,
                    (
                        self.env.action_space.n,  # type: ignore
                        self.env.action_space.n,  # type: ignore
                    ),
                ),
            }
        )

    def observation(self, observation: ObsType) -> Dict[str, np.ndarray]:
        """
        Return a modified observation.

        :param observation: the original observation
        :return: the modified observation
        """
        return {
            self.avail_actions_key: np.eye(
                (
                    self.env.observation_space[
                        self.avail_actions_key
                    ].shape[  # type: ignore
                        0
                    ]
                )
            ),
        }
