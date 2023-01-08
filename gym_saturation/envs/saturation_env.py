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
Saturation Environment
=======================
"""
import random
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import orjson
from gym import Env, spaces

from gym_saturation.clause_space import ClauseSpace
from gym_saturation.utils import FALSEHOOD_SYMBOL, Clause, pretty_print

STATE_DIFF_UPDATED = "state_diff_updated"
PROBLEM_FILENAME = "problem_filename"
MAX_CLAUSES = 100000


class SaturationEnv(Env[dict, int]):
    """
    Saturation algorithm defined in a Reinforcement Learning friendly way.

    .. _saturation_env:

    >>> import os
    >>> import sys
    >>> if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> tptp_folder = files("gym_saturation").joinpath(
    ...     os.path.join("resources", "TPTP-mock")
    ... )
    >>> from glob import glob
    >>> problem_list = sorted(glob(
    ...     os.path.join(tptp_folder, "Problems", "*", "*1-1.p")
    ... ))
    >>> import dataclasses
    >>> class MySaturationEnv(SaturationEnv):
    ...     def reset(
    ...         self,
    ...         *,
    ...         seed: Optional[int] = None,
    ...         return_info: bool = False,
    ...         options: Optional[dict] = None,
    ...     ) -> Union[dict, Tuple[dict, dict]]:
    ...         self._state = {
    ...             "one": Clause(literals="p(X)", label="one"),
    ...             "two": Clause(literals="p(Y)", label="two"),
    ...             "three": Clause(literals="p(Z)", label="three"),
    ...             "four": Clause(literals="~p(X)", label="four")
    ...         }
    ...         return self.state
    ...
    ...     def _do_deductions(self, action: int) -> Tuple[bytes, ...]:
    ...         given_clause = list(self._state.values())[action]
    ...         self._state[given_clause.label] = dataclasses.replace(
    ...             given_clause, processed=True)
    ...         if action == 3:
    ...             self._state["falsehood"] = Clause(
    ...                 literals=FALSEHOOD_SYMBOL, label="falsehood",
    ...                 inference_rule="dummy",
    ...                 inference_parents=("four",)
    ...             )
    ...         return ()
    ...
    >>> env = MySaturationEnv(problem_list)
    >>> env.seed(0)
    0
    >>> len(env.reset()["real_obs"])
    4

    one can look at the current state in TPTP format

    >>> print(env.render())
    cnf(one, lemma, p(X)).
    cnf(two, lemma, p(Y)).
    cnf(three, lemma, p(Z)).
    cnf(four, lemma, ~p(X)).

    ``ansi`` mode returns a JSON representation of the state
    it should be more easily parsable than TPTP, although less human-friendly

    >>> env.render("ansi")
    b'{"one":{"literals":"p(X)","label":"one","role":"lemma","inference_pare...

    other modes are not implemented yet

    >>> env.render(mode="rgb_array")
    Traceback (most recent call last):
     ...
    NotImplementedError

    the test theorem can be proved in three steps

    >>> next_state, reward, done, info = env.step(0)

    ``info`` dict contains the state diff, for example

    >>> info["state_diff_updated"]
    ()

    repeating actions is not allowed

    >>> env.step(0)
    Traceback (most recent call last):
     ...
    ValueError: action 0 is not valid

    there is no reward until the end of an episode

    >>> (reward, done)
    (0.0, False)

    if a proof is found, then reward is ``+1``

    >>> env.step(3)[1:3]
    (1.0, True)

    TSTP proof is now available (one can add ``include`` directive before it
    for validation purposes)

    >>> from gym_saturation.utils import get_tstp_proof
    >>> print(get_tstp_proof(env._state))
    cnf(falsehood, lemma, $false, inference(dummy, [], [four])).

    One can also filter actions relevant to a particular goal:

    >>> from gym_saturation.utils import get_positive_actions
    >>> get_positive_actions(env._state)
    (3, 4)

    the total number of clauses in the state is limited by the ``max_clauses``
    parameter. Let's try setting it and repeating the same solution of the same
    problem:

    >>> env = MySaturationEnv(problem_list, max_clauses=3)
    >>> _ = env.seed(0)
    >>> old_obs = env.reset()

    after the first step we bypass ``max_clauses`` by one, so the episode
    finishes with failure:

    >>> obs, reward, done, _ = env.step(0)
    >>> done, reward
    (True, 0.0)
    >>> env.get_task()
    Traceback (most recent call last):
     ...
    ValueError: Task is not set! Call reset or set_task first.
    >>> env.sample_tasks(1)
    [['.../resources/TPTP-mock/Problems/TST/TST001-1.p']]
    """

    metadata = {"render_modes": ["ansi", "human"]}
    reward_range = (0, 1)

    action_space: spaces.Discrete
    observation_space: spaces.Dict

    def __init__(
        self,
        problem_list: List[str],
        max_clauses: int = MAX_CLAUSES,
    ):
        """
        Initialise spaces et al.

        :param problem_list: a list of the names of TPTP problem files
        :param max_clauses: maximal number of clauses to store in proof state
        """
        super().__init__()
        self.problem_list = problem_list
        self._state: Dict[str, Clause] = {}
        self.action_space = spaces.Discrete(max_clauses)
        self.observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(0, 1, shape=(max_clauses,)),
                "real_obs": ClauseSpace(),
            }
        )
        self.task: Optional[List[str]] = None
        self.problem_filename: Optional[str] = None

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[dict, Tuple[dict, dict]]:  # noqa: D102
        raise NotImplementedError  # pragma: no cover

    def _max_clauses_result(
        self, done: bool, info: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        if not done:
            if len(self._state) > self.action_space.n:
                info.pop(STATE_DIFF_UPDATED)
                return True, info
        return done, info

    @abstractmethod
    def _do_deductions(self, action: int) -> Tuple[Clause, ...]:
        raise NotImplementedError  # pragma: no cover

    def step(self, action: int) -> Tuple[dict, float, bool, Dict[str, Any]]:
        # noqa: D301
        """
        Run one time-step of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info)

        :param action: an action provided by the agent
        :returns: a tuple of four values:\n
            * observation: agent's observation of the current environment
            * reward: amount of reward returned after previous action
            * done: whether the episode has ended, in which case further
              ``step()`` calls will return undefined results
            * info: contains auxiliary diagnostic information (helpful for
              debugging, and sometimes learning)
        :raises ValueError: if the ``action`` identifies an already processed
            clause
        """
        if list(self._state.values())[action].processed:
            raise ValueError(f"action {action} is not valid")
        old_state_size = len(self._state)
        updated = self._do_deductions(action)
        info = {
            STATE_DIFF_UPDATED: updated,
            PROBLEM_FILENAME: self.problem_filename,
        }
        reward, done = (
            (1.0, True)
            if any(
                clause.literals == FALSEHOOD_SYMBOL
                for clause in self._state.values()
            )
            else (
                (old_state_size - len(self._state)) / self.action_space.n,
                False,
            )
        )
        done |= min(
            False if clause.processed is None else clause.processed
            for clause in self._state.values()
        )
        done, info = self._max_clauses_result(done, info)
        return self.state, reward, done, info

    # pylint: disable=inconsistent-return-statements
    def render(self, mode="human"):  # noqa: D102
        if mode == "ansi":
            return orjson.dumps(self._state)
        if mode == "human":
            return "\n".join(
                map(
                    pretty_print,
                    self._state.values(),
                )
            )
        super().render(mode=mode)

    @property
    def state(self) -> dict:
        """Return environment state in Python ``dict`` format."""
        return {
            "real_obs": self._state,
            "action_mask": (
                np.array(
                    [
                        0.0 if clause.processed else 1.0
                        for clause in self._state.values()
                    ]
                    + self.action_space.n * [0.0],
                    np.float32,
                )
            )[: self.action_space.n],
        }

    def seed(self, seed=None):  # noqa: D102
        random.seed(seed)
        return seed

    def sample_tasks(self, n_tasks: int) -> List[List[str]]:
        """
        Sample task of the meta-environment.

        :param n_tasks: number of different TPTP problems needed
        :returns: a list tasks (lists of absolute paths of TPTP problems)
        """
        return [
            [filename]
            for filename in random.sample(self.problem_list, n_tasks)
        ]

    def set_task(self, task: List[str]) -> None:
        """
        Set the specified task to the current environment.

        :param task: a list of absolute paths of TPTP problems
        """
        self.task = task

    def get_task(self) -> List[str]:
        """
        Get the task that the agent is performing in the current environment.

        :returns: a list of absolute paths of TPTP problems
        :raises ValueError: is task is not set
        """
        if self.task:
            return self.task
        raise ValueError("Task is not set! Call reset or set_task first.")
