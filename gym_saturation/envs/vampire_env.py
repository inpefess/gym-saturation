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
Saturation Environment with Vampire back-end
============================================
"""
import dataclasses
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import orjson

from gym_saturation.envs.saturation_env import MAX_CLAUSES, SaturationEnv
from gym_saturation.utils import Clause
from gym_saturation.vampire_wrapper import VampireWrapper


def _parse_vampire_clause(clause_label: str, clause_text: str) -> Clause:
    formula, inference_info = clause_text.split("[")
    pre_inference = inference_info.split("]")[0].split(" ")
    if len(pre_inference) > 1:
        inference_parents = tuple(pre_inference[-1].split(","))
        inference_rule = "_".join(pre_inference[:-1])
    else:
        inference_parents, inference_rule = (), pre_inference[0]
    return Clause(
        literals=formula.strip(),
        label=clause_label,
        inference_rule=inference_rule,
        inference_parents=inference_parents,
        processed=True,
    )


class VampireEnv(SaturationEnv):
    """
    An RL environment around a Vampire prover.

    This class has the same API as SaturationEnv but uses another back-end.
    Here we have only a simple smoke test.

    >>> import sys
    >>> if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> vampire_binary = files("gym_saturation").joinpath(
    ...     os.path.join("resources", "vampire-mock")
    ... )
    >>> vampire_env = VampireEnv(["test"], vampire_binary_path=vampire_binary)
    >>> vampire_env.reset()
    Traceback (most recent call last):
     ...
    ValueError: ('Unexpected response type: ', 'who could expect that?')
    >>> from glob import glob
    >>> problems = sorted(glob(os.path.join(files("gym_saturation").joinpath(
    ...     os.path.join("resources", "TPTP-mock", "Problems")
    ... ), "SET", "*-*.p")))
    >>> vampire_env = VampireEnv(problems)
    >>> observation = vampire_env.reset()
    >>> for action in [0, 3, 6, 7, 8, 9, 10]:
    ...     observation, reward, done, info = vampire_env.step(action)
    >>> print(reward, done)
    1.0 True
    """

    def __init__(
        self,
        problem_list: List[str],
        max_clauses: int = MAX_CLAUSES,
        vampire_binary_path: str = "vampire",
    ):
        """
        Initialise a :ref:`VampireWrapper <vampire-wrapper>`.

        :param problem_list: a list of names of TPTP problem files
        :param max_clauses: maximal number of clauses in proof state
        :param vampire_binary_path: a path to Vampire binary;
            by default we expect it to be in the $PATH
        """
        super().__init__(problem_list, max_clauses)
        self._vampire = VampireWrapper(vampire_binary_path)

    def _parse_vampire_response(
        self, vampire_response: Tuple[Tuple[str, str, str], ...]
    ) -> Dict[str, Clause]:
        updated: Dict[str, Clause] = {}
        for response_type, clause_label, clause_text in vampire_response:
            if response_type in {"new", "final", "input", "fn def discovered"}:
                updated[clause_label] = _parse_vampire_clause(
                    clause_label, clause_text
                )
            elif response_type in {
                "active",
                "forward reduce",
                "passive",
                "backward reduce",
                "new propositional",
            }:
                changed_clause = dataclasses.replace(
                    self._state[clause_label]
                    if clause_label in self._state
                    else updated[clause_label],
                    processed=response_type
                    not in ("passive", "new propositional"),
                )
                updated[clause_label] = changed_clause
            else:
                raise ValueError("Unexpected response type: ", response_type)
        return updated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[Dict, Tuple[dict, dict]]:  # noqa: D102
        self.problem = random.choice(self.problem_list)
        tptp_folder = os.path.join(os.path.dirname(self.problem), "..", "..")
        vampire_response = self._vampire.start(self.problem, tptp_folder)
        self._state = {}
        updated = self._parse_vampire_response(vampire_response)
        self._state = {
            clause.label: dataclasses.replace(clause, birth_step=0)
            for clause in updated.values()
        }
        return self.state

    def _do_deductions(self, action: int) -> Tuple[bytes, ...]:
        given_clause = list(self._state.values())[action]
        updated = self._parse_vampire_response(
            self._vampire.pick_a_clause(given_clause.label)
        )
        self._state.update(updated)
        return tuple(map(orjson.dumps, updated.values()))
