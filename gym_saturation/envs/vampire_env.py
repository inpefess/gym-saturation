# Copyright 2021-2025 Boris Shminke
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
Saturation Environment with Vampire back-end
============================================
"""  # noqa: D205, D400

import os
import re
from typing import Any

from gym_saturation.constants import FALSEHOOD_SYMBOL
from gym_saturation.envs.saturation_env import SaturationEnv
from gym_saturation.vampire_wrapper import VampireWrapper


class VampireEnv(SaturationEnv):
    """
    An RL environment wrapper around Vampire prover.

    :param prover_binary_path: a path to Vampire binary;
        by default we expect it to be in the $PATH

    Refer to :ref:`saturation_env` for more documentation.

    We can run a full Gymnasium environment check:

    >>> from gymnasium.utils.env_checker import check_env
    >>> import gymnasium as gym
    >>> env = gym.make("Vampire-v0").unwrapped
    >>> check_env(env)

    repeating actions change nothing

    >>> env = gym.make("Vampire-v0")
    >>> _ = env.reset()
    >>> one = env.step("c_1")
    >>> two = env.step("c_1")
    >>> one == two
    True

    sometimes Vampire can solve a problem during pre-processing

    >>> from gym_saturation.constants import MOCK_TPTP_PROBLEM
    >>> trivial_problem = os.path.join(os.path.dirname(MOCK_TPTP_PROBLEM),
    ...     "TST002-1.p")
    >>> env.unwrapped.set_task(trivial_problem)
    >>> _, _ = env.reset()
    >>> env.unwrapped._terminated
    True
    >>> _, _, terminated, _, _ = env.step("anything")
    >>> terminated
    True
    >>> env.close()

    a test of an unexpected reply from Vampire

    >>> from gym_saturation.constants import MOCK_TPTP_FOLDER
    >>> vampire_binary = os.path.join(MOCK_TPTP_FOLDER, "..", "vampire-mock")
    >>> vampire_env = VampireEnv(prover_binary_path=vampire_binary)
    >>> vampire_env.reset()
    Traceback (most recent call last):
     ...
    ValueError: ('Unexpected response type: ', 'who could expect that?')
    """

    def __init__(  # noqa: D107
        self,
        prover_binary_path: str = "vampire",
    ):
        super().__init__()
        self._vampire = VampireWrapper(prover_binary_path)

    def _parse_vampire_response(
        self, vampire_response: tuple[tuple[str, str, str], ...]
    ) -> tuple[tuple[str, ...], set[str]]:
        new_labels: set[str] = set()
        new_clauses: tuple[str, ...] = ()
        for response_type, clause_label, clause_text in vampire_response:
            new_label = "c_" + clause_label
            if response_type == "passive" or FALSEHOOD_SYMBOL in clause_text:
                new_clauses += (
                    self._parse_vampire_clause(new_label, clause_text),
                )
                new_labels.add(new_label)
            elif response_type not in {
                "active",
                "forward reduce",
                "backward reduce",
                "new propositional",
                "new",
                "final",
                "input",
                "fn def discovered",
            }:
                raise ValueError("Unexpected response type: ", response_type)
        return new_clauses, new_labels

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        """
        Reset the environment.

        :param seed: seed for compatibility
        :param options: options for compatibility
        :returns: observations and info
        """
        super().reset(seed=seed)
        tptp_folder = os.path.join(
            os.path.dirname(self.get_task()), "..", ".."
        )
        vampire_response = self._vampire.start(self.get_task(), tptp_folder)
        new_clauses, new_labels = self._parse_vampire_response(
            vampire_response
        )
        self._terminated = max(
            (FALSEHOOD_SYMBOL in clause for clause in new_clauses),
            default=False,
        )
        self._available_actions = new_labels
        return new_clauses, {}

    def _do_deductions(self, action: str) -> tuple[tuple[str, ...], set[str]]:
        return self._parse_vampire_response(
            # the first two characters are `c_`
            self._vampire.pick_a_clause(action[2:])
        )

    def _parse_vampire_clause(
        self, clause_label: str, clause_text: str
    ) -> str:
        literals, inference_rule, inference_parents = re.findall(
            r"(.+) \[([^\d,]+)([\d,]*)\]", clause_text
        )[0]
        literals = literals.replace(" ", "")
        inference_rule = inference_rule.strip().replace(" ", "_")
        inference_parents = "c_" + inference_parents.replace(",", ",c_")
        if inference_rule != "input":
            return (
                f"cnf({clause_label},plain,{literals},"
                f"inference({inference_rule},"
                f"[],[{inference_parents}]))."
            )
        return f"cnf({clause_label},axiom,{literals},file('input.p'))."

    def close(self) -> None:
        """Terminate Vampire process."""
        self._vampire.terminate()
