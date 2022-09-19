# Copyright 2021-2022 Boris Shminke
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
import dataclasses
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import orjson
from gym import Env, spaces
from tptp_lark_parser.grammar import Clause
from tptp_lark_parser.tptp_parser import TPTPParser

from gym_saturation.clause_space import ClauseSpace
from gym_saturation.logic_ops.factoring import all_possible_factors
from gym_saturation.logic_ops.paramodulation import all_paramodulants_from_list
from gym_saturation.logic_ops.reflexivity_resolution import (
    all_possible_reflexivity_resolvents,
)
from gym_saturation.logic_ops.resolution import all_possible_resolvents
from gym_saturation.logic_ops.utils import (
    is_tautology,
    reduce_to_proof,
    reindex_variables,
)

STATE_DIFF_UPDATED = "state_diff_updated"
POSITIVE_ACTIONS = "positive_actions"
PROBLEM_FILENAME = "problem_filename"
MAX_CLAUSES = 100000


class SaturationEnv(Env[dict, int]):
    """
    Saturation algorithm defined in a Reinforcement Learning friendly way.

    .. _saturation_env:

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
    >>> env = SaturationEnv(problem_list)
    >>> env.seed(0)
    0
    >>> len(env.reset()["real_obs"])
    4

    one can look at the current state in TPTP format

    >>> print(env.render())
    cnf(this_is_a_test_case_1, hypothesis, p(c)).
    cnf(this_is_a_test_case_2, hypothesis, ~p(c)).
    cnf(test_axiom, axiom, c = d).
    cnf(test_axiom_2, axiom, ~c = e).

    ``ansi`` mode returns a JSON representation of the state
    it should be more easily parsable than TPTP, although less human-friendly

    >>> env.render("ansi")
    [...{"literals":[{"negated":false,"atom":{"index":298,"arguments":[{"ind...

    other modes are not implemented yet

    >>> env.render(mode="rgb_array")
    Traceback (most recent call last):
     ...
    NotImplementedError

    the test theorem can be proved in three steps

    >>> next_state, reward, done, info = env.step(0)

    ``info`` dict contains the state diff, for example

    >>> import json
    >>> json.loads(info["state_diff_updated"][0])["processed"]
    True

    repeating actions is not allowed

    >>> env.step(0)
    Traceback (most recent call last):
     ...
    ValueError: action 0 is not valid

    there is no reward until the end of an episode

    >>> (reward, done)
    (0.0, False)

    if a proof is found, then reward is ``+1``

    >>> env.step(1)[1:3]
    (1.0, True)

    TSTP proof is now available (one can add ``include`` directive before it
    for validation purposes)

    >>> print(env._tptp_parser.cnf_parser.pretty_print(
    ...     env._tptp_parser.parse(env.tstp_proof)[0])
    ... )
    cnf(..., lemma, $false, inference(resolution, [], [this_is_a_test_case_1, this_is_a_test_case_2])).

    the relevant actions are filtered too

    >>> env.positive_actions
    (0, 1, 4)

    the total number of clauses in the state is limited by the ``max_clauses``
    parameter. Let's try setting it and repeating the same solution of the same
    problem:

    >>> env = SaturationEnv(problem_list, max_clauses=3)
    >>> _ = env.seed(0)
    >>> old_obs = env.reset()

    after the first step we bypass ``max_clauses`` by one, so the episode
    finishes with failure:

    >>> obs, reward, done, _ = env.step(0)
    >>> done, reward
    (True, 0.0)
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
        self._state_set: Set[Tuple[bytes, ...]] = set()
        self.action_space = spaces.Discrete(max_clauses)
        self.observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(0, 1, shape=(max_clauses,)),
                "real_obs": ClauseSpace(),
            }
        )
        self.problem: Optional[str] = None
        tptp_folder = os.path.join(
            os.path.dirname(problem_list[0]), "..", ".."
        )
        self._tptp_parser = TPTPParser(tptp_folder, extendable=True)

    def _init_clauses(self) -> Dict[str, Clause]:
        self.problem = random.choice(self.problem_list)
        with open(self.problem, "r", encoding="utf-8") as problem_file:
            problem_text = problem_file.read()
        parsed_clauses = self._tptp_parser.parse(problem_text)
        return {
            clause.label: dataclasses.replace(
                clause,
                birth_step=0,
                inference_parents=(),
                inference_rule=None,
                processed=False,
            )
            for clause in parsed_clauses
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[dict, Tuple[dict, dict]]:  # noqa: D102
        self._state = reindex_variables(self._init_clauses())
        self._state_set = set(
            map(
                lambda clause: tuple(
                    sorted(map(orjson.dumps, clause.literals))
                ),
                self._state.values(),
            )
        )
        return self.state

    def _add_to_state(self, new_clauses: Tuple[Clause, ...]) -> None:
        birth_step = 1 + self.last_birth_step
        for clause in new_clauses:
            if not is_tautology(clause):
                sorted_literals = tuple(
                    sorted(map(orjson.dumps, clause.literals))
                )
                if sorted_literals not in self._state_set:
                    self._state[clause.label] = dataclasses.replace(
                        clause, birth_step=birth_step, processed=False
                    )
                    self._state_set.add(sorted_literals)

    def _do_deductions(self, action: int) -> Tuple[bytes, ...]:
        state_len_before = len(self._state)
        given_clause_label, given_clause = list(self._state.items())[action]
        unprocessed_clauses = tuple(
            clause for clause in self._state.values() if clause.processed
        )
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
        self._state[given_clause_label] = dataclasses.replace(
            given_clause, processed=True
        )
        return tuple(
            map(orjson.dumps, list(self._state.values())[state_len_before:])
        ) + (orjson.dumps(list(self._state.values())[action]),)

    def _proof_found_result(
        self, reward: float, info: Dict[str, Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        if tuple(
            True for clause in self._state.values() if clause.literals == ()
        ):
            info[POSITIVE_ACTIONS] = self.positive_actions
            return 1.0, True, info
        return reward, False, info

    def _max_clauses_result(
        self,
        done: bool,
        info: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        if not done:
            if len(self._state) > self.action_space.n:
                info.pop(STATE_DIFF_UPDATED)
                return True, info
        return done, info

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
        updated = self._do_deductions(action)
        reward = 0.0
        info = {STATE_DIFF_UPDATED: updated, PROBLEM_FILENAME: self.problem}
        reward, done, info = self._proof_found_result(reward, info)
        done |= min(
            False if clause.processed is None else clause.processed
            for clause in self._state.values()
        )
        done, info = self._max_clauses_result(done, info)
        return self.state, reward, done, info

    @property
    def last_birth_step(self) -> int:
        """Return the last birth step number of clauses in the proof state."""
        return max(getattr(clause, "birth_step", 0) for clause in self._state)

    # pylint: disable=inconsistent-return-statements
    def render(self, mode="human"):  # noqa: D102
        if mode == "ansi":
            return self.state["real_obs"]
        if mode == "human":
            return "\n".join(
                map(
                    self._tptp_parser.cnf_parser.pretty_print,
                    self._state.values(),
                )
            )
        super().render(mode=mode)

    @property
    def state(self) -> dict:
        """Return environment state in Python ``dict`` format."""
        return {
            "real_obs": [
                orjson.dumps(clause) for clause in self._state.values()
            ],
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

    @property
    def tstp_proof(self) -> str:
        """Return TSTP proof (if found; raises an error otherwise)."""
        return "\n".join(
            reversed(
                [
                    self._tptp_parser.cnf_parser.pretty_print(clause)
                    for clause in reduce_to_proof(self._state)
                    if clause.inference_rule is not None
                ]
            )
        )

    @property
    def positive_actions(self) -> Tuple[int, ...]:
        """
        Return a sequence of actions which contributed to the proof found.

        If there is no proof yet, raises an error.
        """
        proof = reduce_to_proof(self._state)
        return tuple(
            action
            for action, clause in enumerate(self._state.values())
            if clause in proof
        )
