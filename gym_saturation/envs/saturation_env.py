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
from glob import glob
from typing import List, Optional, Tuple

from gym import Env

from gym_saturation.grammar import Clause
from gym_saturation.logic_ops.resolution import all_possible_resolutions
from gym_saturation.logic_ops.utils import (
    clause_in_a_list,
    is_tautology,
    reindex_variables,
)
from gym_saturation.parsing.json_grammar import ClauseJSONEncoder
from gym_saturation.parsing.tptp_parser import TPTPParser

INFERRED_CLAUSES_PREFIX = "inferred_"


class SaturationEnv(Env):
    """
    saturation algorithm defined in a Reiforcement Learning friendly way

    >>> import sys
    >>> if sys.version_info.major == 3 and sys.version_info.minor == 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> tptp_folder = files("gym_saturation").joinpath("resources/TPTP-mock")
    >>> env = SaturationEnv(
    ...     step_limit=3,
    ...     tptp_folder=tptp_folder
    ... )
    >>> # there is nothing non-deterministic here, but the seed can be set
    >>> env.seed(0)
    0
    >>> # initially, there are only 4 unprocessed clauses
    >>> # problem is not chosen yet
    >>> env.problem
    Traceback (most recent call last):
     ...
    ValueError: Problem no defined. Run env.reset() first
    >>> len(env.reset())
    4
    >>> # now the problem is defined
    >>> print(os.path.basename(env.problem))
    TST001-1.p
    >>> # the test theorem can be proved in three steps
    >>> env.step(0)[1:3]
    (0.0, False)
    >>> # repeating actions is not allowed
    >>> env.step(0)
    Traceback (most recent call last):
     ...
    ValueError: action 0 is not valid
    >>> # there is no reward until the end of an episode
    >>> env.step(1)[1:3]
    (0.0, False)
    >>> # only ``ansi`` rendering method is implemented
    >>> len(env.render("ansi"))
    1422
    >>> env.render()
    Traceback (most recent call last):
     ...
    NotImplementedError
    >>> # if a proof is found, then reward is ``+1``
    >>> env.step(4)[1:3]
    (1.0, True)
    >>> env = SaturationEnv(
    ...     step_limit=1,
    ...     tptp_folder=tptp_folder
    ... )
    >>> # one can also choose a particular problem file during reset
    >>> problem = os.path.join(tptp_folder, "Problems", "TST", "TST001-1.p")
    >>> result = env.reset(problem)
    >>> # if the proof is not found after a fixed number of steps
    >>> # the reward is ``0``
    >>> env.step(0)[1:3]
    (0.0, True)
    """

    def __init__(
        self,
        step_limit: int,
        tptp_folder: str,
    ):
        super().__init__()
        self.step_limit = step_limit
        self.tptp_folder = tptp_folder
        self._step_count = 0
        self._starting_label_index = 0
        self._state: List[Clause] = []
        self._problem: Optional[str] = None

    def _init_clauses(self, filename: str):
        clauses = TPTPParser().parse(
            filename,
            self.tptp_folder,
        )
        for clause in clauses:
            clause.birth_step = 0
            clause.inference_parents = []
            clause.processed = False
        return clauses

    # pylint: disable=arguments-differ
    def reset(self, problem: Optional[str] = None) -> list:
        if problem is None:
            self._problem = random.choice(
                glob(os.path.join(self.tptp_folder, "Problems", "*", "*-*.p"))
            )
        else:
            self._problem = problem
        self._step_count = 0
        self._state = reindex_variables(self._init_clauses(self._problem), "X")
        self._starting_label_index = 0
        self.action_space = list(range(len(self._state)))
        return self.state

    def _do_resolutions(self, given_clause: Clause) -> None:
        if not given_clause.processed:
            new_resolutions = all_possible_resolutions(
                [clause for clause in self._state if clause.processed],
                given_clause,
                INFERRED_CLAUSES_PREFIX,
                self._starting_label_index,
            )
            for new_resolution in new_resolutions:
                if not is_tautology(new_resolution):
                    if not clause_in_a_list(new_resolution, self._state):
                        new_resolution.birth_step = self._step_count
                        new_resolution.processed = False
                        self._state.append(new_resolution)
            self._starting_label_index += len(new_resolutions)

    def step(self, action: int) -> Tuple[list, float, bool, dict]:
        if action not in self.action_space:
            raise ValueError(f"action {action} is not valid")
        self._step_count += 1
        if self._state[action].literals == []:
            self._state[action].processed = True
            return (
                self.state,
                1.0,
                True,
                dict(),
            )
        self._do_resolutions(self._state[action])
        self._state[action].processed = True
        if (
            min(
                [
                    False if clause.processed is None else clause.processed
                    for clause in self._state
                ]
            )
            or self._step_count >= self.step_limit
        ):
            return self.state, 0.0, True, dict()
        self.action_space = [
            i for i, clause in enumerate(self._state) if not clause.processed
        ]
        return (
            self.state,
            0.0,
            False,
            dict(),
        )

    # pylint: disable=inconsistent-return-statements
    def render(self, mode="human"):
        if mode == "ansi":
            return str(self.state)
        super().render(mode=mode)

    @property
    def state(self) -> list:
        """
        :returns: environment state in Python ``dict`` format
        """
        return json.loads(json.dumps(self._state, cls=ClauseJSONEncoder))

    @property
    def problem(self) -> str:
        """
        :returns: full filename of a problem
        """
        if self._problem is not None:
            return self._problem
        raise ValueError("Problem no defined. Run env.reset() first")

    def seed(self, seed=None):
        random.seed(seed)
        return seed
