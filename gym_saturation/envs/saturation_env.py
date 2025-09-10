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
Saturation Environment
=======================
"""  # noqa: D205, D400

import random
from abc import abstractmethod
from typing import Any

from gymnasium import Env, spaces
from gymnasium.spaces.text import alphanumeric

from gym_saturation.constants import FALSEHOOD_SYMBOL, MOCK_TPTP_PROBLEM

ALPHANUMERIC_WITH_UNDERSCORE = "".join(alphanumeric) + "_"
EXTENDED_ALPHANUMERIC = ALPHANUMERIC_WITH_UNDERSCORE + "(), |~=!$.'"


class SaturationEnv(Env[tuple[str, ...], str]):
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
    action_space = spaces.Text(256, charset=ALPHANUMERIC_WITH_UNDERSCORE)
    observation_space = spaces.Sequence(
        spaces.Text(4000, charset=EXTENDED_ALPHANUMERIC)
    )

    def __init__(  # noqa: D107
        self,
    ):
        super().__init__()
        self._task = MOCK_TPTP_PROBLEM
        self._terminated = False
        self._available_actions: set[str] = set()

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
        random.seed(seed)
        self._terminated = False
        self._available_actions = set()
        return (), {}

    @abstractmethod
    def _do_deductions(self, action: Any) -> tuple[tuple[str, ...], set[str]]:
        raise NotImplementedError  # pragma: no cover

    def step(
        self, action: Any
    ) -> tuple[tuple[str, ...], float, bool, bool, dict[str, Any]]:
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
        """  # noqa: D301
        new_clauses: tuple[str, ...] = ()
        if not self._terminated and action in self._available_actions:
            new_clauses, new_actions = self._do_deductions(action)
            self._terminated = max(
                (FALSEHOOD_SYMBOL in clause for clause in new_clauses),
                default=False,
            )
            self._available_actions.discard(action)
            self._available_actions.update(new_actions)
        return (
            new_clauses,
            0.0,
            self._terminated,
            False,
            {},
        )

    def render(self) -> None:  # type: ignore
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
        """
        return self._task
