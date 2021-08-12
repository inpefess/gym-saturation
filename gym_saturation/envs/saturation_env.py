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
from typing import Dict, List, Optional, Tuple

from gym import Env

from gym_saturation.grammar import Clause
from gym_saturation.logic_ops.factoring import all_possible_factors
from gym_saturation.logic_ops.paramodulation import all_paramodulants_from_list
from gym_saturation.logic_ops.reflexivity_resolution import (
    all_possible_reflexivity_resolvents,
)
from gym_saturation.logic_ops.resolution import all_possible_resolvents
from gym_saturation.logic_ops.utils import (
    clause_in_a_list,
    is_tautology,
    reindex_variables,
)
from gym_saturation.parsing.json_grammar import ClauseJSONEncoder
from gym_saturation.parsing.tptp_parser import TPTPParser

INFERRED_CLAUSES_PREFIX = "inferred_"
STATE_DIFF_UPDATED = "state_diff_updated"


class SaturationEnv(Env):
    """
    saturation algorithm defined in a Reiforcement Learning friendly way

    >>> import sys
    >>> if sys.version_info.major == 3 and sys.version_info.minor == 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> tptp_folder = files("gym_saturation").joinpath("resources/TPTP-mock")
    >>> from glob import glob
    >>> problem_list = glob(
    ...     os.path.join(tptp_folder, "Problems", "*", "*1-1.p")
    ... )
    >>> env = SaturationEnv(3, problem_list)
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
    1420
    >>> env.render()
    Traceback (most recent call last):
     ...
    NotImplementedError
    >>> # if a proof is found, then reward is ``+1``
    >>> env.step(4)[1:3]
    (1.0, True)
    >>> env = SaturationEnv(1, problem_list)
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
        problem_list: List[str],
    ):
        super().__init__()
        self.step_limit = step_limit
        self.problem_list = problem_list
        self._step_count = 0
        self._inference_count = 0
        self._state: List[Clause] = []
        self._problem: Optional[str] = None

    def _init_clauses(self):
        tptp_folder = os.path.join(os.path.dirname(self._problem), "..", "..")
        clauses = TPTPParser().parse(
            self._problem,
            tptp_folder,
        )
        for clause in clauses:
            clause.birth_step = 0
            clause.inference_parents = []
            clause.processed = False
        return clauses

    # pylint: disable=arguments-differ
    def reset(self, problem: Optional[str] = None) -> list:
        if problem is None:
            self._problem = random.choice(self.problem_list)
        else:
            self._problem = problem
        self._step_count = 0
        self._state = reindex_variables(self._init_clauses(), "X")
        self.action_space = list(range(len(self._state)))
        return self.state

    def _add_to_state(self, new_clauses: List[Clause]) -> None:
        self._inference_count += len(new_clauses)
        for clause in new_clauses:
            if not is_tautology(clause):
                if not clause_in_a_list(clause, self._state):
                    clause.birth_step = self._step_count
                    clause.processed = False
                    self._state.append(clause)

    def _do_deductions(self, action: int) -> Dict[int, Clause]:
        state_len_before = len(self._state)
        given_clause = self._state[action]
        if not given_clause.processed:
            unprocessed_clauses = [
                clause for clause in self._state if clause.processed
            ]
            self._add_to_state(
                all_possible_resolvents(
                    unprocessed_clauses,
                    given_clause,
                    INFERRED_CLAUSES_PREFIX,
                    self._inference_count,
                )
            )
            self._add_to_state(
                all_paramodulants_from_list(
                    unprocessed_clauses,
                    given_clause,
                    INFERRED_CLAUSES_PREFIX,
                    self._inference_count,
                )
            )
            self._add_to_state(
                all_possible_factors(
                    given_clause,
                    INFERRED_CLAUSES_PREFIX,
                    self._inference_count,
                )
            )
            self._add_to_state(
                all_possible_reflexivity_resolvents(
                    given_clause,
                    INFERRED_CLAUSES_PREFIX,
                    self._inference_count,
                )
            )
        return dict(
            [
                (
                    i + state_len_before,
                    json.loads(json.dumps(clause, cls=ClauseJSONEncoder)),
                )
                for i, clause in enumerate(self._state[state_len_before:])
            ]
            + [
                (
                    action,
                    json.loads(
                        json.dumps(self._state[action], cls=ClauseJSONEncoder)
                    ),
                )
            ]
        )

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
                {STATE_DIFF_UPDATED: {action: self._state[action]}},
            )
        updated = self._do_deductions(action)
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
            return self.state, 0.0, True, {STATE_DIFF_UPDATED: updated}
        self.action_space = [
            i for i, clause in enumerate(self._state) if not clause.processed
        ]
        return (self.state, 0.0, False, {STATE_DIFF_UPDATED: updated})

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
