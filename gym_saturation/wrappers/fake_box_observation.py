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
Fake Box Observation Wrapper
=============================
"""
from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType


class FakeBoxObservation(gym.ObservationWrapper):
    """
    A wrapper which makes an observation a constant dictionary.

    .. _fake_box:

    >>> env = FakeBoxObservation(gym.make("CartPole-v1"))
    >>> observation, info = env.reset()
    >>> observation
    {'item': array([[1., 0.],
           [0., 1.]])}
    """

    observation_space = gym.spaces.Dict({"item": gym.spaces.Box(0, 1, (2, 2))})

    def observation(self, observation: ObsType) -> Dict[str, np.ndarray]:
        """
        Return a modified observation.

        :param observation: the original observation
        :return: the modified observation
        """
        return {"item": np.eye(2)}
