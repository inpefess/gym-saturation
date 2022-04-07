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
"""
Saturation Environment with Vampire backend
============================================
"""
import dataclasses
import os
import random
from typing import Dict, List, Tuple

import orjson

from gym_saturation.envs.saturation_env import MAX_CLAUSES, SaturationEnv
from gym_saturation.grammar import Clause
from gym_saturation.vampire_wrapper import VampireWrapper


class VampireEnv(SaturationEnv):
    """
    an RL environment around a Vampire prover
    saturation algorithm defined in a Reiforcement Learning friendly way

    This class has the same API as SaturationEnv but uses another backend.
    For testing, see ``gym_saturation.agent_testing`` module.
    """

    def __init__(
        self,
        vampire_binary_path: str,
        problem_list: List[str],
        max_clauses: int = MAX_CLAUSES,
    ):
        super().__init__(problem_list, max_clauses)
        self._vampire = VampireWrapper(vampire_binary_path)

    def _parse_vampire_clause(
        self, clause_number: int, clause_text: str
    ) -> Tuple[Clause, ...]:
        formula, inference_info = clause_text.split("[")
        pre_inference = inference_info.split("]")[0].split(" ")
        if len(pre_inference) > 1:
            inference_parents = ",".join(
                [f"x{index}" for index in pre_inference[-1].split(",")]
            )
            inference_rule = "_".join(pre_inference[:-1])
        else:
            inference_parents, inference_rule = "", pre_inference[0]
        return self._tptp_parser.parse(
            f"cnf(x{clause_number}, hypothesis, ({formula}), "
            + f"inference({inference_rule}, [], [{inference_parents}])).",
            "",
        )

    def _add_new_clause(
        self, clause_number: int, clause_text: str
    ) -> Tuple[Clause, ...]:
        expected_next_number = len(self._state) + 1
        if clause_number > expected_next_number:
            raise ValueError(
                "Unexpected order of clauses: ", clause_number, clause_text
            )
        if clause_number < expected_next_number:
            return ()
        new_clause = self._parse_vampire_clause(clause_number, clause_text)
        self._state += new_clause
        return new_clause

    def _parse_vampire_reponse(
        self, vampire_response: Tuple[Tuple[str, int, str], ...]
    ) -> Tuple[Clause, ...]:
        updated: Tuple[Clause, ...] = ()
        for response_type, clause_number, clause_text in vampire_response:
            if response_type in ("new", "final", "input"):
                updated += self._add_new_clause(clause_number, clause_text)
            elif response_type in ("active", "forward reduce"):
                processed_clause = (
                    dataclasses.replace(
                        self._state[clause_number - 1], processed=True
                    ),
                )
                updated += processed_clause
                self._state = (
                    self._state[: clause_number - 1]
                    + processed_clause
                    + self._state[clause_number:]
                )
            elif response_type not in ("passive", "new propositional"):
                raise ValueError("Unexpected reposnse type: ", response_type)
        return updated

    def reset(self) -> dict:
        self.problem = random.choice(self.problem_list)
        tptp_folder = os.path.join(os.path.dirname(self.problem), "..", "..")
        vampire_response = self._vampire.start(self.problem, tptp_folder)
        self._parse_vampire_reponse(vampire_response)
        self._state = tuple(
            dataclasses.replace(clause, birth_step=0, processed=False)
            for clause in self._state
        )
        return self.state

    def _do_deductions(self, action: int) -> Dict[int, bytes]:
        given_clause = self._state[action]
        if not given_clause.processed:
            updated = self._parse_vampire_reponse(
                self._vampire.pick_a_clause(action + 1)
            )
        else:
            updated = ()
        return {
            int(clause.label[1:]) - 1: orjson.dumps(clause)  # type: ignore
            for clause in updated
        }
