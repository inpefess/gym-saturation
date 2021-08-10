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
import sys
from argparse import ArgumentParser, Namespace
from operator import itemgetter
from typing import Callable, List, Optional

from gym_saturation.envs import SaturationEnv
from gym_saturation.logic_ops.utils import clause_length
from gym_saturation.parsing.json_grammar import dict_to_clause


def save_final_state(
    problem_filename: str, output_folder: str, state: list
) -> None:
    """
    save the final state of solving a TPTP problem

    :param problem_filename: a full path to the problem file
    :param output_folder: where to write subfolders and JSON files
    :param state: a JSON-formatted representation of the state
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
        os.path.join(output_subfolder, f"{problem_name}.json"), "w"
    ) as data_file:
        data_file.write(json.dumps(state))


def size_policy(state: list) -> int:
    """
    an example of an implemented policy

    :param state: a list of clauses
    :returns: the index of the clause with minimal length
    """
    return min(
        [
            (i, clause_length(dict_to_clause(clause)))
            for i, clause in enumerate(state)
            if not clause["processed"]
        ],
        key=itemgetter(1),
    )[0]


def episode(
    problem_filename: str,
    output_folder: str,
    step_limit: int,
    policy: Callable,
) -> None:
    """
    tries to solve the problem and log the clauses

    >>> import shutil
    >>> test_policy_output = "test_policy_output"
    >>> shutil.rmtree(test_policy_output, ignore_errors=True)
    >>> import sys
    >>> if sys.version_info.major == 3 and sys.version_info.minor == 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> problem_filename = (
    ...     files("gym_saturation")
    ...     .joinpath("resources/TPTP-mock/Problems/TST/TST001-1.p")
    ... )
    >>> episode(problem_filename, test_policy_output, 5, size_policy)
    >>> with open(os.path.join(test_policy_output, "TST/TST001-1.json")) as f:
    ...     text = f.read()
    >>> len(text)
    1419

    :param problem_filename: the name of a problem file
    :param output_folder: where to log given clause
    :param step_limit: a maximal number of steps in an episode
    :returns:
    """
    env = SaturationEnv(step_limit, [problem_filename])
    state = env.reset(problem=problem_filename)
    done = False
    while not done:
        action = policy(state)
        state, _, done, _ = env.step(action)
    save_final_state(problem_filename, output_folder, env.state)


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
    argument_parser.add_argument("--problem_filename", type=str, required=True)
    argument_parser.add_argument("--output_folder", type=str, required=True)
    argument_parser.add_argument("--step_limit", type=int, required=True)
    parsed_args = argument_parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    arguments = parse_args()
    episode(
        arguments.problem_filename,
        arguments.output_folder,
        arguments.step_limit,
        size_policy,
    )
