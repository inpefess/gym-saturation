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
Saturation Environment
=======================
"""
import random
from abc import abstractmethod
from typing import Any, Optional

from gymnasium import Env, spaces
from gymnasium.spaces.text import alphanumeric

from gym_saturation.constants import MOCK_TPTP_PROBLEM
from gym_saturation.proof_state import ProofState

ALPHANUMERIC_WITH_UNDERSCORE = "".join(alphanumeric) + "_"
SHORT_TEXT_SPACE = spaces.Text(256, charset=ALPHANUMERIC_WITH_UNDERSCORE)
LONG_TEXT_SPACE = spaces.Text(
    4000,
    charset=ALPHANUMERIC_WITH_UNDERSCORE + "(), |~=!$",
)


class SaturationEnv(Env[tuple[dict[str, Any], ...], str]):
    """
    Saturation algorithm in a reinforcement learning friendly way.

    It's an abstract class, so here we have only trivial smoke tests.
    One should override ``_do_deductions`` method in children classes.

    Refer to :ref:`saturation_env` for more documentation.

    >>> class DummyProver(SaturationEnv):
    ...     def _do_deductions(action):
    ...         pass

    >>> env = DummyProver()
    """

    reward_range = (0, 1)
    action_space = spaces.Text(256)
    observation_space: spaces.Sequence

    def __init__(
        self,
    ):
        """Initialise spaces et al."""
        super().__init__()
        self.state = ProofState(clauses={})
        self.observation_space = spaces.Sequence(
            spaces.Dict(
                {
                    "label": SHORT_TEXT_SPACE,
                    "role": SHORT_TEXT_SPACE,
                    "literals": LONG_TEXT_SPACE,
                    "inference_rule": SHORT_TEXT_SPACE,
                    "inference_parents": spaces.Sequence(SHORT_TEXT_SPACE),
                }
            )
        )
        self._task = MOCK_TPTP_PROBLEM

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        """
        Reset the environment.

        :param seed: seed for compatibility
        :param options: options for compatibility
        :returns: observations and info
        """
        super().reset(seed=seed)
        random.seed(seed)
        self.state = ProofState(clauses={})
        return tuple(self.state.clauses.values()), {}

    @abstractmethod
    def _do_deductions(self, action: Any) -> None:
        raise NotImplementedError  # pragma: no cover

    def step(
        self, action: Any
    ) -> tuple[tuple[dict[str, Any], ...], float, bool, bool, dict[str, Any]]:
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
            * terminated: whether the proof was found
            * truncated: whether the episode was finished for an external
              reason (e.g. time limit)
            * info: contains auxiliary diagnostic information (helpful for
              debugging, and sometimes learning)
        """
        if not self.state.terminated:
            self._do_deductions(action)
        return (
            tuple(self.state.clauses.values()),
            1.0 if self.state.terminated else 0.0,
            self.state.terminated,
            False,
            {},
        )

    def render(self) -> None:
        """No render."""

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
