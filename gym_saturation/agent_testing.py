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
import json
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple

import gym

from gym_saturation.envs import SaturationEnv
from gym_saturation.envs.saturation_env import STATE_DIFF_UPDATED
from gym_saturation.grammar import Clause
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


def save_final_state(
    problem_filename: str, output_folder: str, episode_memory: List[Transition]
) -> None:
    """
    save the final environment state of solving a TPTP problem

    :param problem_filename: a full path to the problem file
    :param output_folder: where to write subfolders and JSON files
    :param episode_memory: a list of transitions during the episode
    :returns:
    """
    problem_name = os.path.splitext(os.path.basename(problem_filename))[0]
    output_subfolder = os.path.join(
        output_folder, os.path.basename(os.path.dirname(problem_filename))
    )
    try:
        os.mkdir(output_subfolder)
    except FileExistsError:
        pass
    with open(
        os.path.join(output_subfolder, f"{problem_name}.json"),
        "w",
        encoding="utf-8",
    ) as data_file:
        data_file.write(json.dumps(episode_memory[-1].observation["real_obs"]))


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
    >>> agents = [SizeAgent(), RandomAgent(), AgeAgent()]
    >>> for i in range(3):
    ...     env = gym.make(
    ...         "gym_saturation:saturation-v0",
    ...         step_limit=5,
    ...         problem_list=[problem_list[i]],
    ...     )
    ...     save_final_state(
    ...         problem_list[i],
    ...         test_agent_output,
    ...         episode(env, agents[i])
    ...     )
    >>> print(sorted(agent_testing_report(
    ...     problem_list + ["this_is_a_test_case"], test_agent_output
    ... ).items()))
    [('TST001-1', ('PROOF_FOUND', 2, 2)), ('TST002-1', ('STEP_LIMIT', 5, -1)), ('TST003-1', ('STEP_LIMIT', 4, -1)), ('this_is_a_test_case', ('ERROR', -1, -1))]

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


def _analyse_proof(env_state: List[Clause]) -> Tuple[str, int, int]:
    empty_clauses = [
        clause
        for clause in env_state
        if clause.literals == [] and clause.processed
    ]
    step_count = max(
        [
            -1 if clause.birth_step is None else clause.birth_step
            for clause in env_state
        ]
    )
    if len(empty_clauses) > 1:
        return "ERROR", -1, -1
    if len(empty_clauses) == 0:
        return "STEP_LIMIT", step_count, -1
    return (
        "PROOF_FOUND",
        step_count,
        proof_length(empty_clauses[0], env_state),
    )


def proof_length(empty_clause: Clause, env_state: List[Clause]) -> int:
    """

    :param empty_clause: a clause with no literals (final step of a proof)
    :param env_state: a list of clauses
    :returns: the number of steps in a refutation proof
    """
    proof_parts = (
        []
        if empty_clause.inference_parents is None
        else empty_clause.inference_parents
    )
    proof = [empty_clause]
    while proof_parts != []:
        label = proof_parts.pop()
        proof += [clause for clause in env_state if clause.label == label]
        proof_parts += (
            []
            if proof[-1].inference_parents is None
            else proof[-1].inference_parents
        )
    return len({clause.birth_step for clause in proof})


def agent_testing_report(
    problem_list: List[str], testing_result_folder: str
) -> Dict[str, Tuple[str, int, int]]:
    """

    :param problem_list: a list of full paths to problem files
    :param testing_result_folder: a folder filled by runs of ``episode``
        function over ``problem_list``
    :returns: a dictionary with problems as keys, and the following values:
        * status (PROOF_FOUND, STEP_LIMIT, ERROR)
        * step count (``-1`` for errors)
        * proof length (``-1`` for no proof)
    """
    res = {}
    for problem in problem_list:
        problem_name = os.path.splitext(os.path.basename(problem))[0]
        result_filename = os.path.join(
            testing_result_folder,
            os.path.basename(os.path.dirname(problem)),
            f"{problem_name}.json",
        )
        if os.path.exists(result_filename):
            with open(result_filename, "r", encoding="utf-8") as json_file:
                env_state = [
                    dict_to_clause(clause)
                    for clause in json.loads(json_file.read())
                ]
            res[problem_name] = _analyse_proof(env_state)
        else:
            res[problem_name] = ("ERROR", -1, -1)
    return res


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    arguments = parse_args()
    environment = gym.make(
        "gym_saturation:saturation-v0",
        step_limit=arguments.step_limit,
        problem_list=[arguments.problem_filename],
    )
    an_episode_memory = episode(environment, SizeAgent())
    save_final_state(
        arguments.problem_filename, arguments.output_folder, an_episode_memory
    )
