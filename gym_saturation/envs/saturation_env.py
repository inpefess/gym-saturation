# Copyright 2021-2023 Boris Shminke
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
import random
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env, spaces
from gymnasium.spaces.text import alphanumeric

from gym_saturation.constants import MOCK_TPTP_PROBLEM
from gym_saturation.proof_state import ProofState
from gym_saturation.utils import pretty_print

MAX_CLAUSES = 1000
ALPHANUMERIC_WITH_UNDERSCORE = "".join(alphanumeric) + "_"
SHORT_TEXT_SPACE = spaces.Text(256, charset=ALPHANUMERIC_WITH_UNDERSCORE)
LONG_TEXT_SPACE = spaces.Text(
    4000,
    charset=ALPHANUMERIC_WITH_UNDERSCORE + "(), |~=!$",
)


class SaturationEnv(Env[Tuple[Dict[str, Any], ...], np.int64]):
    """
    Saturation algorithm in a reinforcement learning friendly way.

    It's an abstract class, so here we have only trivial smoke tests.
    One should override ``_do_deductions`` method in children classes.

    Refer to :ref:`saturation_env` for more documentation.

    >>> class DummyProver(SaturationEnv):
    ...     def _do_deductions(action):
    ...         pass
    >>> env = DummyProver(render_mode="rgb_array")
    Traceback (most recent call last):
     ...
    ValueError: Expected a render mode among ['ansi', 'human'] but got rgb_a...

    >>> env = DummyProver()
    >>> env.render_mode = "rgb_array"
    >>> env.render()
    Traceback (most recent call last):
     ...
    NotImplementedError
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 1}
    reward_range = (0, 1)
    action_space: spaces.Space
    observation_space: spaces.Sequence

    def __init__(
        self,
        max_clauses: int = MAX_CLAUSES,
        render_mode: str = "human",
    ):
        """
        Initialise spaces et al.

        :param max_clauses: maximal number of clauses to store in proof state
        :param render_mode: a mode of running ``render`` method
        """
        super().__init__()
        self.state = ProofState(
            clauses={},
            step_number=-1,
            max_clauses=max_clauses,
        )
        self.observation_space = spaces.Sequence(
            spaces.Dict(
                {
                    "label": SHORT_TEXT_SPACE,
                    "role": SHORT_TEXT_SPACE,
                    "literals": LONG_TEXT_SPACE,
                    "inference_rule": SHORT_TEXT_SPACE,
                    "inference_parents": spaces.Sequence(SHORT_TEXT_SPACE),
                    "birth_step": spaces.Discrete(max_clauses),
                }
            )
        )
        self._task = MOCK_TPTP_PROBLEM
        self.render_mode = self._check_render_mode(render_mode)

    def _check_render_mode(self, render_mode: str) -> str:
        if render_mode in self.metadata["render_modes"]:
            return render_mode
        raise ValueError(
            f"Expected a render mode among {self.metadata['render_modes']} "
            f"but got {render_mode}"
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Any]]:
        """
        Reset the environment.

        :param seed: seed for compatibility
        :param options: options for compatibility
        :returns: observations and info
        """
        super().reset(seed=seed)
        random.seed(seed)
        self.state = ProofState(
            clauses={},
            step_number=0,
            max_clauses=self.state.max_clauses,
        )
        return tuple(self.state.clauses.values()), {}

    @abstractmethod
    def _do_deductions(self, action: Any) -> None:
        raise NotImplementedError  # pragma: no cover

    def step(
        self, action: Any
    ) -> Tuple[Tuple[Dict[str, Any], ...], float, bool, bool, Dict[str, Any]]:
        # noqa: D301
        """
        Run one time-step of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.
        Accepts an action and returns a tuple
        (observation, reward, terminated, truncated, info)

        :param action: an action provided by the agent
        :returns: a tuple of four values:\n
            * observation: agent's observation of the current environment
            * reward: amount of reward returned after previous action
            * terminated: Whether the proof was found
            * truncated: Whether the maximal number of clauses in the proof
              state were reached
            * info: contains auxiliary diagnostic information (helpful for
              debugging, and sometimes learning)
        """
        if not (self.state.terminated or self.state.truncated):
            self.state.step_number += 1
            self._do_deductions(action)
        if self.state.truncated:
            self.on_truncated()
        return (
            tuple(self.state.clauses.values()),
            1.0 if self.state.terminated else 0.0,
            self.state.terminated,
            self.state.truncated,
            {},
        )

    # pylint: disable=inconsistent-return-statements
    def render(self) -> None:
        """
        Return or print the proof state.

        :returns: proof state (TPTP formatted) or nothing
            (depending on ``render_mode``)
        """
        tptp_string = "\n".join(
            map(
                pretty_print,
                self.state.clauses.values(),
            )
        )
        if self.render_mode == "ansi":
            return tptp_string  # type: ignore
        if self.render_mode == "human":
            print(tptp_string)
        else:
            super().render()

    def set_task(self, task: str) -> None:
        """
        Set the specified task to the current environment.

        :param task: a TPTP problem filename
        """
        self._task = task

    def get_task(self) -> str:
        """
        Get the task that the agent is performing in the current environment.

        :returns: a TPTP problem filename
        :raises ValueError: is task is not set
        """
        return self._task

    def on_truncated(self) -> None:
        """Prover-specific episode truncation."""
