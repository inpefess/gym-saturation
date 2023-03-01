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
An examples of Thompson sampling
=================================
"""
import os
from typing import Any, Dict

import gymnasium as gym
from ray.rllib.algorithms.bandit import BanditLinTSConfig
from ray.tune.registry import register_env

from gym_saturation.wrappers.age_weight_bandit import AgeWeightBandit
from gym_saturation.wrappers.fake_box_observation import FakeBoxObservation


def env_creator(env_config: Dict[str, Any]) -> gym.Env:
    """
    Return a multi-armed-bandit version of a saturation prover.

    :param env_config: an environment config
    :returns: an environment
    """
    return FakeBoxObservation(
        AgeWeightBandit(gym.make("Vampire-v0", **env_config))
    )


def train_thompson_sampling() -> None:
    """Train Thompson sampling."""
    register_env("VampireBandit", env_creator)
    problem_list = [
        os.path.join(
            os.environ["WORK"],
            "data",
            "TPTP-v8.1.2",
            "Problems",
            "GRP",
            "GRP001-1.p",
        )
    ]
    config = (
        BanditLinTSConfig()
        .rollouts(num_rollout_workers=10)
        .environment(
            env_config={"max_clauses": 300, "problem_list": problem_list}
        )
    )
    algo = config.build(env="VampireBandit")
    for _ in range(100):
        algo.train()


if __name__ == "__main__":
    train_thompson_sampling()
