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
from typing import Any, Callable, Dict, List, Optional, Tuple

from gym_saturation.envs import SaturationEnv
from gym_saturation.grammar import Clause
from gym_saturation.logic_ops.utils import clause_length
from gym_saturation.parsing.json_grammar import dict_to_clause


@dataclass
class Transition:
    """
    an object describing environment's state before the agent's action, the
    action itself and its results
    """

    state: List[Dict[str, Any]]
    action: int
    policy_info: Dict[str, Any]
    next_state: List[Dict[str, Any]]
    reward: float
    done: bool
    env_info: Dict[str, Any]


def save_final_state(
    problem_filename: str, output_folder: str, episode_memory: List[Transition]
) -> None:
    """
    save the final state of solving a TPTP problem

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
        data_file.write(json.dumps(episode_memory[-1].next_state))


# pylint: disable=unused-argument
def size_policy(
    state: list, policy_info: Dict[str, Any], env_info: Dict[str, Any]
) -> Tuple[int, Dict[str, Any]]:
    """
    an example of an implemented policy

    :param state: a list of clauses
    :param policy_info: added for compatibility purpose
    :param env_info: added for compatibility purpose
    :returns: the index of the clause with minimal length, empty info dict
    """
    return (
        min(
            [
                (i, clause_length(dict_to_clause(clause)))
                for i, clause in enumerate(state)
                if not clause["processed"]
            ],
            key=itemgetter(1),
        )[0],
        {},
    )


# pylint: disable=unused-argument
def age_policy(
    state: list, policy_info: Dict[str, Any], env_info: Dict[str, Any]
) -> Tuple[int, Dict[str, Any]]:
    """
    another example of an implemented policy

    :param state: a list of clauses
    :param policy_info: added for compatibility purpose
    :param env_info: added for compatibility purpose
    :returns: the index of the first not yet processed clause, empty info dict
    """
    return (
        min([i for i, clause in enumerate(state) if not clause["processed"]]),
        {},
    )


# pylint: disable=unused-argument
def random_policy(
    state: list, policy_info: Dict[str, Any], env_info: Dict[str, Any]
) -> Tuple[int, Dict[str, Any]]:
    """
    the most basic RL policy --- only exploration

    :param state: a list of clauses
    :param policy_info: added for compatibility purpose
    :param env_info: added for compatibility purpose
    :returns: (an index of a randomly selected not yet processed clause,
        empty info dict)
    """
    return (
        random.choice(
            [i for i, clause in enumerate(state) if not clause["processed"]]
        ),
        {},
    )


def episode(
    problem_filename: str,
    step_limit: int,
    policy: Callable[
        [List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]],
        Tuple[int, Dict[str, Any]],
    ],
) -> Tuple[SaturationEnv, List[Transition]]:
    """
    tries to solve the problem and logs the clauses

    >>> import shutil
    >>> test_policy_output = "test_policy_output"
    >>> shutil.rmtree(test_policy_output, ignore_errors=True)
    >>> os.mkdir(test_policy_output)
    >>> import sys
    >>> if sys.version_info.major == 3 and sys.version_info.minor == 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> from glob import glob
    >>> problem_list = glob(os.path.join(
    ...     files("gym_saturation")
    ...     .joinpath("resources/TPTP-mock/Problems/TST")
    ... , "*-*.p"))
    >>> for i in range(2):
    ...     save_final_state(
    ...         problem_list[i],
    ...         test_policy_output,
    ...         episode(problem_list[i], 5, size_policy)[1]
    ...     )
    >>> print(sorted(policy_testing_report(
    ...     problem_list + ["this_is_a_test_case"], test_policy_output
    ... ).items()))
    [('TST001-1', ('PROOF_FOUND', 2, 2)), ('TST002-1', ('STEP_LIMIT', 5, -1)), ('this_is_a_test_case', ('ERROR', -1, -1))]

    :param problem_filename: the name of a problem file
    :param step_limit: a maximal number of steps in an episode
    :param policy: a function, getting state as an argument and returning
        action and info dict
    :returns: the episode memory
    """
    env = SaturationEnv(step_limit, [problem_filename])
    state, done = env.reset(), False
    episode_memory = []
    policy_info: Dict[str, Any] = {}
    env_info: Dict[str, Any] = {}
    while not done:
        action, policy_info = policy(state, policy_info, env_info)
        next_state, reward, done, env_info = env.step(action)
        episode_memory.append(
            Transition(
                state, action, policy_info, next_state, reward, done, env_info
            )
        )
        state = next_state
    return env, episode_memory


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


def _analyse_proof(state: List[Clause]) -> Tuple[str, int, int]:
    empty_clauses = [
        clause
        for clause in state
        if clause.literals == [] and clause.processed
    ]
    step_count = max(
        [
            -1 if clause.birth_step is None else clause.birth_step
            for clause in state
        ]
    )
    if len(empty_clauses) > 1:
        return ("ERROR", -1, -1)
    if len(empty_clauses) == 0:
        return ("STEP_LIMIT", step_count, -1)
    return (
        "PROOF_FOUND",
        step_count,
        proof_length(empty_clauses[0], state),
    )


def proof_length(empty_clause: Clause, state: List[Clause]) -> int:
    """

    :param empty_clause: a clause with no literals (final step of a proof)
    :param state: a list of clauses
    :returns: the number of steps in a refutational proof
    """
    proof_parts = (
        []
        if empty_clause.inference_parents is None
        else empty_clause.inference_parents
    )
    proof = [empty_clause]
    while proof_parts != []:
        label = proof_parts.pop()
        proof += [clause for clause in state if clause.label == label]
        proof_parts += (
            []
            if proof[-1].inference_parents is None
            else proof[-1].inference_parents
        )
    return len({clause.birth_step for clause in proof})


def policy_testing_report(
    problem_list: List[str], testing_result_folder: str
) -> Dict[str, Tuple[str, int, int]]:
    """

    :param problem_list: a list of full paths to problem files
    :param testing_result_folder: a folder filled by runs of ``episode``
        function over ``problem_list``
    :returns: a dictionary with problems as kets, and the following values:
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
                state = [
                    dict_to_clause(clause)
                    for clause in json.loads(json_file.read())
                ]
            res[problem_name] = _analyse_proof(state)
        else:
            res[problem_name] = ("ERROR", -1, -1)
    return res


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    arguments = parse_args()
    random.seed(hash(arguments.problem_filename))
    _, an_episode_memory = episode(
        arguments.problem_filename,
        arguments.step_limit,
        size_policy,
    )
    save_final_state(
        arguments.problem_filename, arguments.output_folder, an_episode_memory
    )
