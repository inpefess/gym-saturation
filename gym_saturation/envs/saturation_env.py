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
    reduce_to_proof,
    reindex_variables,
)
from gym_saturation.parsing.json_grammar import clause_to_dict
from gym_saturation.parsing.tptp_parser import TPTPParser, clause_to_tptp

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

    there is nothing non-deterministic here, but the seed can be set

    >>> env.seed(0)
    0

    problem is not chosen yet

    >>> env.problem
    Traceback (most recent call last):
     ...
    ValueError: Problem not defined. Run env.reset() first
    >>> len(env.reset())
    4

    now the problem is defined

    >>> print(os.path.basename(env.problem))
    TST001-1.p

    one can look at the current state in TPTP format

    >>> print(env.render())
    cnf(this_is_a_test_case_1, hypothesis, this_is_a_test_case(test_constant) ).
    cnf(this_is_a_test_case_2, hypothesis, ~this_is_a_test_case(test_constant) ).
    cnf(test_axiom, hypothesis, =(test_constant,X0) ).
    cnf(test_axiom_2, hypothesis, ~=(test_constant,0) ).

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

    >>> print(env.tstp_proof)
    cnf(inferred_0, hypothesis, $false, inference(resolution, [], [this_is_a_test_case_1,this_is_a_test_case_2])).

    >>> env = SaturationEnv(1, problem_list)

    one can also choose a particular problem file during reset

    >>> problem = os.path.join(tptp_folder, "Problems", "TST", "TST001-1.p")
    >>> result = env.reset(problem)

    if the proof is not found after a fixed number of steps the reward is ``0``

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
        with open(self._problem, "r", encoding="utf-8") as problem_file:
            problem_text = problem_file.read()
        clauses = TPTPParser().parse(problem_text, tptp_folder)
        for clause in clauses:
            clause.birth_step = 0
            clause.inference_parents = []
            clause.inference_rule = None
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
        self._state[action].processed = True
        return dict(
            [
                (i + state_len_before, clause_to_dict(clause))
                for i, clause in enumerate(self._state[state_len_before:])
            ]
            + [(action, clause_to_dict(self._state[action]))]
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
                {
                    STATE_DIFF_UPDATED: {
                        action: clause_to_dict(self._state[action])
                    }
                },
            )
        updated = self._do_deductions(action)
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
        if mode == "human":
            return "\n".join(
                [clause_to_tptp(clause) for clause in self._state]
            )
        super().render(mode=mode)

    @property
    def state(self) -> list:
        """
        :returns: environment state in Python ``dict`` format
        """
        return clause_to_dict(self._state)

    @property
    def problem(self) -> str:
        """
        :returns: full filename of a problem
        """
        if self._problem is not None:
            return self._problem
        raise ValueError("Problem not defined. Run env.reset() first")

    def seed(self, seed=None):
        random.seed(seed)
        return seed

    @property
    def tstp_proof(self) -> str:
        """
        :returns: TSTP proof (if found; raises error otherwise)
        """
        return "\n".join(
            [
                clause_to_tptp(clause)
                for clause in reduce_to_proof(self._state)
                if clause.inference_rule is not None
            ]
        )
