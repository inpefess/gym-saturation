"""
Copyright 2021 Boris Shminke

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import random
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Dict, List, Optional

import gym

from gym_saturation.envs import SaturationEnv
from gym_saturation.envs.saturation_env import STATE_DIFF_UPDATED
from gym_saturation.logic_ops.utils import clause_length
from gym_saturation.parsing.json_grammar import dict_to_clause


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
    """agent which selects the shortest clause"""

    def get_action(
        self,
        observation: dict,
        reward: float,
        info: Dict[str, Any],
    ) -> int:
        return min(
            [
                (i, clause_length(dict_to_clause(clause)))
                for i, clause in enumerate(observation["real_obs"])
                if not clause["processed"]
            ],
            key=itemgetter(1),
        )[0]


class AgeAgent(BaseAgent):
    """agent which selects the oldest clause"""

    def get_action(
        self,
        observation: dict,
        reward: float,
        info: Dict[str, Any],
    ) -> int:
        return min(
            [
                i
                for i, clause in enumerate(observation["real_obs"])
                if not clause["processed"]
            ]
        )


class SizeAgeAgent(BaseAgent):
    """
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
                for i, clause in enumerate(observation["real_obs"])
                if not clause["processed"]
            ]
        )


def episode(env: SaturationEnv, agent: BaseAgent) -> List[Transition]:
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
    ...     .joinpath("resources/TPTP-mock/Problems/TST")
    ... , "*-*.p")))
    >>> random.seed(0)
    >>> agents = [SizeAgeAgent(2, 1), SizeAgeAgent(1, 2), RandomAgent()]
    >>> for i in range(3):  # doctest: +ELLIPSIS
    ...     env = gym.make(
    ...         "gym_saturation:saturation-v0",
    ...         step_limit=5,
    ...         problem_list=[problem_list[i]],
    ...     )
    ...     agent_testing_report(env, agents[i])
    Proof of length 1 found in 4 steps:
    cnf(_0, hypothesis, $false, inference(resolution, [], [this_is_a_test_case_1, this_is_a_test_case_2])).
    Step limit reached
    Proof of length 2 found in 5 steps:
      ...
    cnf(_2, hypothesis, $false, inference(resolution, [], [p, _1])).

    :param env: a `gym_saturation` environment
    :param agent: an initialized agent. Must have `get_action` method
    :param problem_filename: the name of a problem file
    :returns: the episode memory
    """
    env_state, reward, done = env.reset(), 0.0, False
    episode_memory = []
    info: Dict[str, Any] = {STATE_DIFF_UPDATED: dict(enumerate(env_state))}
    while not done:
        action = agent.get_action(env_state, reward, info)
        observation, reward, done, info = env.step(action)
        episode_memory.append(
            Transition(
                env_state,
                agent.state,
                action,
                observation,
                reward,
                done,
                info,
            )
        )
        env_state = observation
    return episode_memory


def parse_args(args: Optional[List[str]] = None) -> Namespace:
    """
    >>> parse_args([
    ...     "--problem_filename", "test",
    ...     "--output_folder", "this_is_a_test_case",
    ...     "--step_limit", "1"
    ... ])
    Namespace(output_folder='this_is_a_test_case', problem_filename='test', step_limit=1)

    :param args: a list of string arguments
        (for testing and use in a non script scenario)
    :returns: arguments namespace for the script
    """
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--output_folder", type=str, required=True)
    argument_parser.add_argument("--problem_filename", type=str, required=True)
    argument_parser.add_argument("--step_limit", type=int, required=True)
    parsed_args = argument_parser.parse_args(args)
    return parsed_args


def agent_testing_report(env: SaturationEnv, agent: BaseAgent) -> None:
    """
        print a report after testing an agent in an environment

    :param env: an environment
    :param agent: an agent
    :returns:
    """
    an_episode_memory = episode(env, agent)
    if an_episode_memory[-1].reward == 1.0:
        a_proof = env.tstp_proof
        proof_length = len(a_proof.split("\n"))
        print(
            f"Proof of length {proof_length} found "
            f"in {len(an_episode_memory)} steps:"
        )
        print(a_proof)
    else:
        print("Step limit reached")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    arguments = parse_args()
    environment = gym.make(
        "gym_saturation:saturation-v0",
        step_limit=arguments.step_limit,
        problem_list=[arguments.problem_filename],
    )
    print(arguments.problem_filename)
    agent_testing_report(environment, SizeAgeAgent(5, 1))
