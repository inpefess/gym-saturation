# Copyright 2021-2022 Boris Shminke

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
There is only one environment in this module.
It's registered using a limit for steps in an episode
"""
from gym.envs.registration import register

register(
    id="GymSaturation-v0",
    entry_point="gym_saturation.envs:SaturationEnv",
    max_episode_steps=1000,
    reward_threshold=1.0,
)
__version__ = "0.3.0"
