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
        self, clause_label: str, clause_text: str
    ) -> Clause:
        formula, inference_info = clause_text.split("[")
        pre_inference = inference_info.split("]")[0].split(" ")
        if len(pre_inference) > 1:
            inference_parents = ",".join(pre_inference[-1].split(","))
            inference_rule = "_".join(pre_inference[:-1])
        else:
            inference_parents, inference_rule = "", pre_inference[0]
        parsed_clause = self._tptp_parser.parse(
            f"cnf({clause_label}, hypothesis, ({formula}), "
            + f"inference({inference_rule}, [], [{inference_parents}])).",
            "",
        )[0]
        return dataclasses.replace(parsed_clause, processed=True)

    def _parse_vampire_reponse(
        self, vampire_response: Tuple[Tuple[str, str, str], ...]
    ) -> Dict[str, Clause]:
        updated: Dict[str, Clause] = {}
        for response_type, clause_label, clause_text in vampire_response:
            if response_type in ("new", "final", "input"):
                updated[clause_label] = self._parse_vampire_clause(
                    clause_label, clause_text
                )
            elif response_type in (
                "active",
                "forward reduce",
                "passive",
                "backward reduce",
            ):
                changed_clause = dataclasses.replace(
                    self._state[clause_label]
                    if clause_label in self._state
                    else updated[clause_label],
                    processed=response_type != "passive",
                )
                updated[clause_label] = changed_clause
            elif response_type not in ("new propositional"):
                raise ValueError("Unexpected reposnse type: ", response_type)
        return updated

    def reset(self) -> dict:
        self.problem = random.choice(self.problem_list)
        tptp_folder = os.path.join(os.path.dirname(self.problem), "..", "..")
        vampire_response = self._vampire.start(self.problem, tptp_folder)
        updated = self._parse_vampire_reponse(vampire_response)
        self._state = {
            clause.label: dataclasses.replace(clause, birth_step=0)
            for clause in updated.values()
        }
        return self.state

    def _do_deductions(self, action: int) -> Tuple[bytes, ...]:
        given_clause = list(self._state.values())[action]
        if not given_clause.processed:
            updated = self._parse_vampire_reponse(
                self._vampire.pick_a_clause(given_clause.label)
            )
        else:
            updated = {}
        self._state.update(updated)
        return tuple(map(orjson.dumps, updated.values()))
