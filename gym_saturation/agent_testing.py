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
Agent Testing
==============
This module is an example of testing your own trained agent.
"""
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from gym_saturation.envs.saturation_env import (
    ACTION_MASK,
    MAX_CLAUSES,
    REAL_OBS,
    SaturationEnv,
)
from gym_saturation.utils import FALSEHOOD_SYMBOL, get_tstp_proof


class BaseAgent(ABC):
    """A basic RL agent."""

    _state: Any = None

    @property
    def state(self):
        """Agent can have its inner state."""
        return self._state

    @abstractmethod
    def get_action(
        self,
        observation: Dict[str, Any],
        reward: float,
        info: Dict[str, Any],
    ) -> int:
        """
        Get an action given observations and other inputs.

        :param observation: an observation returned by the environment after
            the latest action (or it's initial state)
        :param reward: reward from the previous step
            (use zero for the first step)
        :param info: info dictionary returned by the environment
        :returns: an action
        """


class WeightAgent(BaseAgent):
    """
    Agent which selects the shortest clause.

    .. _weight_agent:
    """

    def __init__(self):
        """Don't compute the formulae lengths twice."""
        self._state: Dict[str, Tuple[float, int]] = {}

    def update_state(self, observation: Dict[str, Any]) -> None:
        """
        Update the state of the agent according to the transition.

        :param observation: observation returned by prover
        """
        self._state.update(
            {
                clause["label"]: (len(clause["literals"]), action_mask)
                for clause, action_mask in zip(
                    observation[REAL_OBS],
                    observation[ACTION_MASK].tolist(),
                )
            }
        )

    def get_action(
        self,
        observation: Dict[str, Any],
        reward: float,
        info: Dict[str, Any],
    ) -> int:  # noqa: D102
        self.update_state(observation)
        return min(
            (
                (key, value[0])
                for key, value in enumerate(self.state.values())
                if value[1] == 1.0
            ),
            key=itemgetter(1),
        )[0]


class AgeAgent(BaseAgent):
    """
    Agent which selects the oldest clause.

    .. _age_agent:
    """

    def get_action(
        self,
        observation: Dict[str, Any],
        reward: float,
        info: Dict[str, Any],
    ) -> int:  # noqa: D102
        return observation[ACTION_MASK].argmax()


class AgeWeightAgent(BaseAgent):
    """
    Agent taking several times the smallest clause and then the oldest.

    .. _age_weight_agent:
    """

    def __init__(self, age_steps: int, weight_steps: int):
        """
        Initialise two sub-agents.

        :param age_steps: how many times to select the oldest clause
        :param weight_steps: how many times to select the shortest clause
        """
        self.weight_steps = weight_steps
        self.age_steps = age_steps
        self._step_count = 0
        self._use_weight = False
        self._weight_agent = WeightAgent()
        self._age_agent = AgeAgent()

    def get_action(
        self,
        observation: Dict[str, Any],
        reward: float,
        info: Dict[str, Any],
    ) -> int:  # noqa: D102
        self._step_count += 1
        if self._use_weight:
            if self._step_count >= self.weight_steps:
                self._step_count = 0
                self._use_weight = False
            return self._weight_agent.get_action(observation, reward, info)
        if self._step_count >= self.age_steps:
            self._step_count = 0
            self._use_weight = True
        self._weight_agent.update_state(observation)
        return self._age_agent.get_action(observation, reward, info)


class RandomAgent(BaseAgent):
    """Agent which selects clauses randomly."""

    def get_action(
        self,
        observation: Dict[str, Any],
        reward: float,
        info: Dict[str, Any],
    ) -> int:  # noqa: D102
        return np.random.choice(np.nonzero(observation[ACTION_MASK])[0])


def _proof_found_before_the_start(
    env: SaturationEnv,
) -> Tuple[float, bool, bool]:
    if tuple(
        1
        for clause in env.state.clauses
        if clause["literals"] == FALSEHOOD_SYMBOL
    ):
        return 1.0, True, False
    return 0.0, False, False


def episode(env: SaturationEnv, agent: BaseAgent) -> Tuple[float, bool, int]:
    """
    Try to solve the problem and logs the clauses.

    >>> env = gym.make(
    ...     "Vampire-v0",
    ...     max_clauses=5,
    ... )
    >>> agent_testing_report(env, RandomAgent())
    Proof state size limit reached in ... step(s).

    :param env: a `gym_saturation` environment
    :param agent: an initialised agent. Must have `get_action` method
    :returns: gain, truncation, the number of steps
    """
    obs, info = env.reset()
    reward, terminated, truncated = _proof_found_before_the_start(env)
    gain = reward
    step_count = 0
    while not terminated and not truncated:
        action = agent.get_action(obs, reward, info)
        obs, reward, terminated, truncated, info = env.step(np.int64(action))
        gain += reward
        step_count += 1
    return gain, truncated, step_count


def parse_args(args: Optional[List[str]] = None) -> Namespace:
    """
    Parse script arguments.

    >>> parse_args([
    ...     "--problem_filename", "this_is_a_test_case",
    ... ])
    Namespace(max_clauses=1000, problem_filename='this_is_a_test_case')

    :param args: a list of string arguments
        (for testing and use in a non script scenario)
    :returns: arguments namespace for the script
    """
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "--max_clauses", type=int, required=False, default=MAX_CLAUSES
    )
    argument_parser.add_argument("--problem_filename", type=str, required=True)
    parsed_args = argument_parser.parse_args(args)
    return parsed_args


def agent_testing_report(env: SaturationEnv, agent: BaseAgent) -> None:
    """
    Print a report after testing an agent in an environment.

    :param env: an environment
    :param agent: an agent
    """
    _, truncated, step_count = episode(env, agent)
    if not truncated:
        a_proof = get_tstp_proof(tuple(env.state.clauses))
        proof_length = len(a_proof.split("\n"))
        print(
            f"Proof of length {proof_length} found "
            f"in {step_count} step(s):\n{a_proof}"
        )
    else:
        print(f"Proof state size limit reached in {step_count} step(s).")


def test_agent(args: Optional[List[str]] = None) -> None:
    """
    The main function for this module.

    >>> from gym_saturation.constants import MOCK_TPTP_PROBLEM
    >>> test_agent(["--problem_filename", MOCK_TPTP_PROBLEM])
    Problem file: ...TST001-1.p
    Proof of length 10 found in 6 step(s):
    ...
    cnf(20, lemma, $false, inference(subsumption_resolution, [], [19, 5])).
    >>> import os
    >>> trivial_problem = os.path.join(os.path.dirname(MOCK_TPTP_PROBLEM),
    ...     "TST002-1.p")
    >>> test_agent(["--problem_filename", trivial_problem])
    Problem file: ...TST002-1.p
    Proof of length 3 found in 0 step(s):
     ...
    cnf(3, lemma, $false, inference(subsumption_resolution, [], [2, 1])).

    :param args: parameters from the command line or explicitly set
    """
    arguments = parse_args(args)
    env: SaturationEnv = gym.make(  # type: ignore
        "Vampire-v0",
        max_clauses=arguments.max_clauses,
    )
    env.set_task(arguments.problem_filename)
    print(f"Problem file: {arguments.problem_filename}")
    agent_testing_report(env, AgeWeightAgent(1, 1))  # type: ignore


if __name__ == "__main__":
    test_agent()  # pragma: no cover
