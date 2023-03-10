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
Observation wrapper doubling a key
====================================
"""
from typing import Any, Dict

import gymnasium as gym


class DuplicateKeyObsWrapper(gym.ObservationWrapper):
    """
    Adding a key duplicating an existing one in dictionary observations.

    >>> tptp_folder = getfixture("mock_tptp_folder")  # noqa: F821
    >>> import os
    >>> problem_list = [os.path.join(
    ...     tptp_folder, "Problems", "TST", "TST001-1.p"
    ... )]
    >>> from gym_saturation.envs.dummy_saturation_env import DummySaturationEnv
    >>> env = DuplicateKeyObsWrapper(
    ...     DummySaturationEnv(problem_list=problem_list),
    ...     key_to_duplicate="action_mask",
    ...     new_key="test"
    ... )
    >>> obs, _ = env.reset()
    >>> import numpy as np
    >>> np.array_equal(obs["test"], obs["action_mask"])
    True
    """

    def __init__(self, env: gym.Env, key_to_duplicate: str, new_key: str):
        """Initialise the observation wrapper."""
        super().__init__(env)
        self.key_to_duplicate = key_to_duplicate
        self.new_key = new_key
        old_space: gym.spaces.Dict = self.env.observation_space  # type:ignore
        obs_dict = dict(old_space.items())
        obs_dict.update({self.new_key: old_space[self.key_to_duplicate]})
        self.env.observation_space = gym.spaces.Dict(obs_dict)  # type:ignore

    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        """
        Return a modified observation.

        :param observation: the original observation
        :returns: the modified observation
        """
        new_observation: Dict[str, Any] = observation.copy()  # type: ignore
        new_observation[self.new_key] = new_observation[self.key_to_duplicate]
        return new_observation  # type: ignore
