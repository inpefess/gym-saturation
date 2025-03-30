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

# noqa: D205, D400
"""
Saturation Environment with Vampire back-end
============================================
"""
import os
from typing import Any, Optional

from gym_saturation.constants import FALSEHOOD_SYMBOL
from gym_saturation.envs.saturation_env import SaturationEnv
from gym_saturation.vampire_wrapper import VampireWrapper


class VampireEnv(SaturationEnv):
    """
    An RL environment wrapper around Vampire prover.

    Refer to :ref:`saturation_env` for more documentation.

    We can run a full Gymnasium environment check:

    >>> from gymnasium.utils.env_checker import check_env
    >>> import gymnasium as gym
    >>> env = gym.make("Vampire-v0").unwrapped
    >>> check_env(env)

    repeating actions change nothing

    >>> env = gym.make("Vampire-v0")
    >>> _ = env.reset()
    >>> one = env.step("1")
    >>> two = env.step("1")
    >>> one == two
    True

    sometimes Vampire can solve a problem during pre-processing

    >>> from gym_saturation.constants import MOCK_TPTP_PROBLEM
    >>> trivial_problem = os.path.join(os.path.dirname(MOCK_TPTP_PROBLEM),
    ...     "TST002-1.p")
    >>> env.unwrapped.set_task(trivial_problem)
    >>> _, _ = env.reset()
    >>> env.unwrapped.state.terminated
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

    def __init__(
        self,
        prover_binary_path: str = "vampire",
    ):
        """
        Initialise a :ref:`VampireWrapper <vampire-wrapper>`.

        :param prover_binary_path: a path to Vampire binary;
            by default we expect it to be in the $PATH
        """
        super().__init__()
        self._vampire = VampireWrapper(prover_binary_path)

    def _parse_vampire_response(
        self, vampire_response: tuple[tuple[str, str, str], ...]
    ) -> None:
        for response_type, clause_label, clause_text in vampire_response:
            if response_type == "passive" or FALSEHOOD_SYMBOL in clause_text:
                self.state.clauses[clause_label] = self._parse_vampire_clause(
                    clause_label, clause_text
                )
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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
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
        self._parse_vampire_response(vampire_response)
        return tuple(map(str, self.state.clauses.values())), {}

    def _do_deductions(self, action: str) -> None:
        if action in self.state.clauses:
            self._parse_vampire_response(self._vampire.pick_a_clause(action))

    def _parse_vampire_clause(
        self, clause_label: str, clause_text: str
    ) -> dict[str, Any]:
        formula, inference_info = clause_text.split("[")
        pre_inference = inference_info.split("]")[0].split(" ")
        if len(pre_inference) > 1:
            inference_parents = tuple(pre_inference[-1].split(","))
            inference_rule = "_".join(pre_inference[:-1])
        else:
            inference_parents, inference_rule = (), pre_inference[0]
        return {
            "literals": formula.strip(),
            "label": clause_label,
            "role": "lemma",
            "inference_rule": inference_rule,
            "inference_parents": inference_parents,
        }

    def close(self) -> None:
        """Terminate Vampire process."""
        self._vampire.terminate()
