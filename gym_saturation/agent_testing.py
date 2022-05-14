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
Agent Testing
==============

This module is an example of testing your own trained agent.
"""
import random
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

import gym
import orjson
from gym.wrappers import TimeLimit

from gym_saturation.envs.saturation_env import (
    STATE_DIFF_UPDATED,
    SaturationEnv,
)
from gym_saturation.logic_ops.utils import (
    WrongRefutationProofError,
    clause_length,
)


class BaseAgent:
    """a basic RL agent"""

    _state = None

    @property
    def state(self):
        """agent can have its inner state"""
        return self._state

    def get_action(
        self,
        observation: dict,
        reward: float,
        info: Dict[str, Any],
    ) -> int:
        """
        :param observation: an observation returned by the environment after
            the latest action (or it's initial state)
        :param reward: reward from the previous step
            (use zero for the first step)
        :param info: info dictionary returned by the environment
        :returns: an action
        """


@dataclass
class Transition:
    """
    an object describing environment's and agent's states
    before the agent's action, the action itself and its results
    """

    env_state: dict
    agent_state: Any
    action: int
    observation: dict
    reward: float
    done: bool
    info: Dict[str, Any]


class SizeAgent(BaseAgent):
    """
    .. _size_agent:

    agent which selects the shortest clause
    """

    def __init__(self):
        self._state: Dict[str, Tuple[float, bool]] = {}

    def update_state(self, info: Dict[str, Any]) -> None:
        """
        update the state of the agent according with the transition

        :param info: an info dict (parm of environment response)
        :returns:
        """
        parsed_state_diff = tuple(map(orjson.loads, info[STATE_DIFF_UPDATED]))
        self._state.update(
            {
                clause["label"]: (clause_length(clause), clause["processed"])
                for clause in parsed_state_diff
            }
        )

    def get_action(
        self,
        observation: dict,
        reward: float,
        info: Dict[str, Any],
    ) -> int:
        self.update_state(info)
        return min(
            (
                (key, value[0])
                for key, value in enumerate(self.state.values())
                if not value[1]
            ),
            key=itemgetter(1),
        )[0]


class AgeAgent(BaseAgent):
    """
    .. _age_agent:

    agent which selects the oldest clause
    """

    def get_action(
        self,
        observation: dict,
        reward: float,
        info: Dict[str, Any],
    ) -> int:
        return min(
            [
                i
                for i, clause in enumerate(
                    map(orjson.loads, observation["real_obs"])
                )
                if not clause["processed"]
            ]
        )


class SizeAgeAgent(BaseAgent):
    """
    .. _size_age_agent:

    agent which takes several times the smallest clause and then several
    times the oldest
    """

    def __init__(self, size_steps: int, age_steps: int):
        self.size_steps = size_steps
        self.age_steps = age_steps
        self._step_count = 0
        self._use_size = True
        self._size_agent = SizeAgent()
        self._age_agent = AgeAgent()

    def get_action(
        self,
        observation: dict,
        reward: float,
        info: Dict[str, Any],
    ) -> int:
        self._step_count += 1
        if self._use_size:
            if self._step_count >= self.size_steps:
                self._step_count = 0
                self._use_size = False
            return self._size_agent.get_action(observation, reward, info)
        if self._step_count >= self.age_steps:
            self._step_count = 0
            self._use_size = True
        self._size_agent.update_state(info)
        return self._age_agent.get_action(observation, reward, info)


class RandomAgent(BaseAgent):
    """agent which selects clauses randomly"""

    def get_action(
        self,
        observation: dict,
        reward: float,
        info: Dict[str, Any],
    ) -> int:
        return random.choice(
            [
                i
                for i, clause in enumerate(
                    map(orjson.loads, observation["real_obs"])
                )
                if not clause["processed"]
            ]
        )


def _proof_found_before_the_start(env: SaturationEnv) -> Tuple[float, bool]:
    try:
        _ = env.tstp_proof
        return 1.0, True
    except WrongRefutationProofError:
        return 0.0, False


def episode(env: SaturationEnv, agent: BaseAgent) -> float:
    """
    tries to solve the problem and logs the clauses

    >>> import os
    >>> import shutil
    >>> test_agent_output = "test_agent_output"
    >>> shutil.rmtree(test_agent_output, ignore_errors=True)
    >>> os.mkdir(test_agent_output)
    >>> import sys
    >>> if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> from glob import glob
    >>> problem_list = sorted(glob(os.path.join(
    ...     files("gym_saturation")
    ...     .joinpath(os.path.join(
    ...         "resources", "TPTP-mock", "Problems", "TST"
    ...     ))
    ... , "*-*.p")))
    >>> random.seed(0)
    >>> agents = (SizeAgeAgent(2, 1), SizeAgeAgent(1, 2), RandomAgent())
    >>> max_clauses = (100, 100, 4)
    >>> for i in range(3):  # doctest: +ELLIPSIS
    ...     env = gym.make(
    ...         "GymSaturation-v0",
    ...         problem_list=[problem_list[i]],
    ...         max_clauses=max_clauses[i]
    ...     )
    ...     env._max_episode_steps = 5
    ...     agent_testing_report(env, agents[i])
    Proof of length 1 found in 2 steps:
    cnf(..., lemma, $false, inference(resolution, [], [this_is_a_test_case_1, this_is_a_test_case_2])).
    Step limit reached
    Proof state size limit reached

    :param env: a `gym_saturation` environment
    :param agent: an initialized agent. Must have `get_action` method
    :param problem_filename: the name of a problem file
    :returns: total reward
    """
    obs = env.reset()
    info: Dict[str, Any] = {STATE_DIFF_UPDATED: obs["real_obs"]}
    reward, done = _proof_found_before_the_start(env)
    total_reward = reward
    while not done:
        action = agent.get_action(obs, reward, info)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward


def parse_args(args: Optional[List[str]] = None) -> Namespace:
    """
    >>> parse_args([
    ...     "--problem_filename", "test",
    ...     "--step_limit", "1"
    ... ])
    Namespace(problem_filename='test', step_limit=1, vampire_binary_path=None)

    :param args: a list of string arguments
        (for testing and use in a non script scenario)
    :returns: arguments namespace for the script
    """
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--problem_filename", type=str, required=True)
    argument_parser.add_argument("--step_limit", type=int, required=True)
    argument_parser.add_argument(
        "--vampire_binary_path", type=str, required=False
    )
    parsed_args = argument_parser.parse_args(args)
    return parsed_args


def agent_testing_report(env: SaturationEnv, agent: BaseAgent) -> None:
    """
        print a report after testing an agent in an environment

    :param env: an environment
    :param agent: an agent
    :returns:
    """
    total_reward = episode(env, agent)
    step_count = getattr(env, "_elapsed_steps")
    try:
        a_proof = env.tstp_proof
        if total_reward == 1.0:
            proof_length = len(a_proof.split("\n"))
            print(
                f"Proof of length {proof_length} found "
                f"in {step_count} steps:\n{a_proof}"
            )
    except WrongRefutationProofError:
        print(
            "Step limit reached"
            if step_count == getattr(env, "_max_episode_steps")
            else "Proof state size limit reached"
        )


def test_agent(args: Optional[List[str]] = None) -> None:
    """
    the main function for this module

    >>> if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> import os
    >>> from glob import glob
    >>> problem_filenames = sorted(glob(os.path.join(
    ...     files("gym_saturation")
    ...     .joinpath(os.path.join(
    ...         "resources", "TPTP-mock", "Problems", "TST", "TST00*-1.p"
    ...     ))
    ... )))
    >>> test_agent([
    ...     "--problem_filename", problem_filenames[0],
    ...     "--step_limit", "3"
    ... ])  # doctest: +ELLIPSIS
    Problem file: ...TST001-1.p
    Proof of length 1 found in 2 steps:
    cnf(..., lemma, $false, inference(resolution, [], [this_is_a_test_case_1, this_is_a_test_case_2])).
    >>> for problem_filename in problem_filenames:
    ...     test_agent([
    ...         "--problem_filename", problem_filename,
    ...         "--step_limit", "10",
    ...         "--vampire_binary_path", "vampire",
    ...     ])  # doctest: +ELLIPSIS
    Problem file: ...TST001-1.p
    Proof of length 6 found in 0 steps:
    ...
    Problem file: ...TST002-1.p
    Proof of length 10 found in 4 steps:
    ...
    Problem file: ...TST003-1.p
    Proof of length 5 found in 3 steps:
    ...
    cnf(5, hyp..., $false, inference(subsumption_resolution, [], [4, 3])).
    """
    sys.setrecursionlimit(10000)
    arguments = parse_args(args)
    if arguments.vampire_binary_path is not None:
        env = gym.make(
            "GymVampire-v0",
            problem_list=[arguments.problem_filename],
            vampire_binary_path=arguments.vampire_binary_path,
        )
    else:
        env = gym.make(
            "GymSaturation-v0",
            problem_list=[arguments.problem_filename],
        )
    environment = TimeLimit(env, arguments.step_limit)
    print(f"Problem file: {arguments.problem_filename}")
    agent_testing_report(environment, SizeAgeAgent(5, 1))


if __name__ == "__main__":
    test_agent()  # pragma: no cover
