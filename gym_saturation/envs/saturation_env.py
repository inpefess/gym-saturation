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
Saturation Environment
=======================
"""
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from gym import Env, spaces

from gym_saturation.clause_space import ClauseSpace
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
    reduce_to_proof,
    reindex_variables,
)
from gym_saturation.parsing.json_grammar import clause_to_dict
from gym_saturation.parsing.tptp_parser import TPTPParser

STATE_DIFF_UPDATED = "state_diff_updated"
MAX_CLAUSES = 100000


class SaturationEnv(Env):
    """
    saturation algorithm defined in a Reiforcement Learning friendly way

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
    >>> env = SaturationEnv(problem_list, 1)
    >>> env.seed(0)
    0
    >>> env.reset()
    Traceback (most recent call last):
     ...
    ValueError: Too many clauses: 4
    consider increasing `max_clauses` parameter of `__init__`
    >>> env = SaturationEnv(problem_list)
    >>> env.seed(0)
    0
    >>> len(env.reset()["real_obs"])
    4

    one can look at the current state in TPTP format

    >>> print(env.render())
    cnf(this_is_a_test_case_1, hypothesis, this_is_a_test_case(test_constant)).
    cnf(this_is_a_test_case_2, hypothesis, ~this_is_a_test_case(test_constant)).
    cnf(test_axiom, axiom, test_constant = X0).
    cnf(test_axiom_2, axiom, ~test_constant = 0).

    ``ansi`` mode returns a JSON representation of the state
    it should be more easily parsable than TPTP, although less human-friendly

    >>> env.render("ansi")[:73]
    "[{'class': 'Clause', 'literals': [{'class': 'Literal', 'negated': False, "

    other modes are not implemented yet

    >>> env.render(mode="rgb_array")
    Traceback (most recent call last):
     ...
    NotImplementedError

    the test theorem can be proved in three steps

    >>> next_state, reward, done, info = env.step(0)

    ``info`` dict contains the state diff, for example

    >>> info["state_diff_updated"][0]["processed"]
    True

    repeating actions is not allowed

    >>> env.step(0)
    Traceback (most recent call last):
     ...
    ValueError: action 0 is not valid

    there is no reward until the end of an episode

    >>> env.step(1)[1:3]
    (0.0, False)

    if a proof is found, then reward is ``+1``

    >>> env.step(4)[1:3]
    (1.0, True)

    TSTP proof is now available (one can add ``include`` directive before it
    for validation purposes)

    >>> print(TPTPParser().parse(env.tstp_proof, "")[0])  # doctest: +ELLIPSIS
    cnf(..., lemma, $false(), inference(resolution, [], [this_is_a_test_case_1, this_is_a_test_case_2])).
    """

    def __init__(
        self,
        problem_list: List[str],
        max_clauses: int = MAX_CLAUSES,
    ):
        super().__init__()
        self.problem_list = problem_list
        self._state: List[Clause] = []
        self.action_space = spaces.Discrete(max_clauses)
        self.observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(0, 1, shape=(max_clauses,)),
                "real_obs": ClauseSpace(),
            }
        )

    def _init_clauses(self):
        problem = random.choice(self.problem_list)
        tptp_folder = os.path.join(os.path.dirname(problem), "..", "..")
        with open(problem, "r", encoding="utf-8") as problem_file:
            problem_text = problem_file.read()
        clauses = TPTPParser().parse(problem_text, tptp_folder)
        for clause in clauses:
            clause.birth_step = 0
            clause.inference_parents = []
            clause.inference_rule = None
            clause.processed = False
        return clauses

    def reset(self) -> dict:
        self._state = reindex_variables(self._init_clauses(), "X")
        return self.state

    def _add_to_state(self, new_clauses: List[Clause]) -> None:
        birth_step = 1 + max(
            [getattr(clause, "birth_step", 0) for clause in self._state]
        )
        for clause in new_clauses:
            if not is_tautology(clause):
                if not clause_in_a_list(clause, self._state):
                    clause.birth_step = birth_step
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
                )
            )
            self._add_to_state(
                all_paramodulants_from_list(
                    unprocessed_clauses,
                    given_clause,
                )
            )
            self._add_to_state(
                all_possible_factors(
                    given_clause,
                )
            )
            self._add_to_state(
                all_possible_reflexivity_resolvents(
                    given_clause,
                )
            )
        self._state[action].processed = True
        return dict(
            [
                (i + state_len_before, clause_to_dict(clause))
                for i, clause in enumerate(self._state[state_len_before:])
            ]
            + [(action, clause_to_dict(self._state[action]))]
        )

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        if self._state[action].processed:
            raise ValueError(f"action {action} is not valid")
        if self._state[action].literals == []:
            self._state[action].processed = True
            return (
                self.state,
                1.0,
                True,
                {
                    STATE_DIFF_UPDATED: {
                        action: clause_to_dict(self._state[action])
                    }
                },
            )
        updated = self._do_deductions(action)
        if min(
            [
                False if clause.processed is None else clause.processed
                for clause in self._state
            ]
        ):
            return self.state, 0.0, True, {STATE_DIFF_UPDATED: updated}
        return (
            self.state,
            0.0,
            False,
            {STATE_DIFF_UPDATED: updated},
        )

    # pylint: disable=inconsistent-return-statements
    def render(self, mode="human"):
        if mode == "ansi":
            return str(self.state["real_obs"])
        if mode == "human":
            return "\n".join(map(str, self._state))
        super().render(mode=mode)

    @property
    def state(self) -> dict:
        """
        :returns: environment state in Python ``dict`` format
        """
        if len(self._state) > self.action_space.n:
            raise ValueError(
                f"Too many clauses: {len(self._state)}\n"
                "consider increasing `max_clauses` parameter of `__init__`"
            )
        return {
            "real_obs": clause_to_dict(self._state),
            "action_mask": (
                np.array(
                    [
                        0.0 if clause.processed else 1.0
                        for clause in self._state
                    ]
                    + self.action_space.n * [0.0],
                    np.float32,
                )
            )[: self.action_space.n],
        }

    def seed(self, seed=None):
        random.seed(seed)
        return seed

    @property
    def tstp_proof(self) -> str:
        """
        :returns: TSTP proof (if found; raises error otherwise)
        """
        return "\n".join(
            reversed(
                [
                    str(clause)
                    for clause in reduce_to_proof(self._state)
                    if clause.inference_rule is not None
                ]
            )
        )
