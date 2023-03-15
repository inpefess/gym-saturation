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
import argparse
import os
from typing import Any, Dict

import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.bandit import BanditLinTSConfig
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.sample_batch import SampleBatch
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


class PatchedRandomPolicy(RandomPolicy):
    """RandomPolicy from Ray examples misses a couple of methods."""

    # pylint: disable=unused-argument, missing-param-doc
    def load_batch_into_buffer(
        self, batch: SampleBatch, buffer_index: int = 0
    ) -> int:
        """Don't load anything anywhere."""
        return 0

    # pylint: disable=unused-argument
    def learn_on_loaded_batch(
        self, offset: int = 0, buffer_index: int = 0
    ) -> dict:
        """Don't learn anything and return empty results."""
        return {}


# pylint: disable=too-few-public-methods
class RandomAlgorithm(Algorithm):
    """Algorithm taking random actions and not learning anything."""

    # pylint: disable=unused-argument, missing-param-doc
    @classmethod
    def get_default_policy_class(cls, config: AlgorithmConfig) -> RandomPolicy:
        """We created PatchedRandomPolicy exactly for this algorithm."""
        return PatchedRandomPolicy


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random_baseline",
        action="store_true",
        help="Run random baseline instead of Thompson sampling",
    )
    return parser.parse_args()


def train_thompson_sampling() -> None:
    """Train Thompson sampling."""
    args = parse_args()
    register_env("VampireBandit", env_creator)
    problem_list = [
        os.path.join(
            os.environ["WORK"],
            "data",
            "TPTP-v8.1.2",
            "Problems",
            "SET",
            "SET001-1.p",
        )
    ]
    if args.random_baseline:
        algo = (
            AlgorithmConfig(RandomAlgorithm)
            .framework("torch")
            .environment(
                "VampireBandit",
                env_config={"max_clauses": 20, "problem_list": problem_list},
            )
        ).build()
    else:
        algo = (
            BanditLinTSConfig()
            .environment(
                env_config={"max_clauses": 20, "problem_list": problem_list}
            )
            .build(env="VampireBandit")
        )
    for _ in range(20):
        algo.train()


if __name__ == "__main__":
    train_thompson_sampling()
