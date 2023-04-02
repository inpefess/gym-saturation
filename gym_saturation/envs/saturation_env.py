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

from gym_saturation.proof_state import ProofState
from gym_saturation.utils import (
    FALSEHOOD_SYMBOL,
    MOCK_TPTP_PROBLEM,
    pretty_print,
)

MAX_CLAUSES = 1000
ALPHANUMERIC_WITH_UNDERSCORE = "".join(alphanumeric) + "_"
REAL_OBS = "real_obs"
ACTION_MASK = "action_mask"
SHORT_TEXT_SPACE = spaces.Text(256, charset=ALPHANUMERIC_WITH_UNDERSCORE)
LONG_TEXT_SPACE = spaces.Text(
    4000,
    charset=ALPHANUMERIC_WITH_UNDERSCORE + "(), |~=!$",
)


class SaturationEnv(Env[Dict[str, Any], np.int64]):
    """
    Saturation algorithm in a reinforcement learning friendly way.

    It's an abstract class, so here we have only trivial smoke tests.
    One should override ``_do_deductions`` method in children classes.

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
    action_space: spaces.Discrete
    observation_space: spaces.Dict

    def __init__(
        self,
        max_clauses: int = MAX_CLAUSES,
        render_mode: str = "human",
    ):
        """
        Initialise spaces et al.

        :param problem_list: a list of the names of TPTP problem files
        :param max_clauses: maximal number of clauses to store in proof state
        """
        super().__init__()
        self.state = ProofState(
            clauses=[],
            clause_labels=[],
            action_mask=np.zeros((max_clauses,), dtype=np.int8),
            step_number=-1,
        )
        self.action_space = spaces.Discrete(max_clauses)
        self.observation_space = spaces.Dict(
            {
                REAL_OBS: spaces.Sequence(
                    spaces.Dict(
                        {
                            "label": SHORT_TEXT_SPACE,
                            "role": SHORT_TEXT_SPACE,
                            "literals": LONG_TEXT_SPACE,
                            "inference_rule": SHORT_TEXT_SPACE,
                            "inference_parents": spaces.Sequence(
                                SHORT_TEXT_SPACE
                            ),
                            "birth_step": spaces.Discrete(max_clauses),
                        }
                    )
                ),
                ACTION_MASK: spaces.Box(0, 1, (max_clauses,), dtype=np.int8),
            }
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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # noqa: D102
        super().reset(seed=seed)
        random.seed(seed)
        self.state = ProofState(
            clauses=[],
            clause_labels=[],
            action_mask=np.zeros_like(self.state.action_mask),
            step_number=0,
        )
        return {
            REAL_OBS: tuple(self.state.clauses),
            ACTION_MASK: self.state.action_mask,
        }, {}

    @abstractmethod
    def _do_deductions(self, action: np.int64) -> None:
        raise NotImplementedError  # pragma: no cover

    def step(
        self, action: np.int64
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
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
        :raises ValueError: if the ``action`` identifies an already processed
            clause
        """
        if self.state.action_mask[action] == 0.0:
            raise ValueError(f"action {action} is not valid")
        self.state.step_number += 1
        self._do_deductions(action)
        self.state.action_mask[action] = 0.0
        truncated = len(self.state.clauses) > int(self.action_space.n)
        terminated = (
            max(
                clause["literals"] == FALSEHOOD_SYMBOL
                for clause in self.state.clauses
            )
            or self.state.action_mask.max() == 0.0
        ) and not truncated
        reward = 1.0 if terminated else 0.0
        return (
            {
                REAL_OBS: tuple(self.state.clauses),
                ACTION_MASK: self.state.action_mask,
            },
            reward,
            terminated,
            truncated,
            {},
        )

    # pylint: disable=inconsistent-return-statements
    def render(self):  # noqa: D102
        tptp_string = "\n".join(
            map(
                pretty_print,
                self.state.clauses,
            )
        )
        if self.render_mode == "ansi":
            return tptp_string
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
