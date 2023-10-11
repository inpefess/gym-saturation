#   Copyright 2023 Boris Shminke
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# noqa: D205, D400
"""
Useful Actions Wrapper
======================
"""
from typing import Any, Dict, Tuple

from gymnasium.core import ActionWrapper


class UsefulActionsWrapper(ActionWrapper):
    """
    A wrapper returning negative reward in observation doesn't change.

    >>> import gymnasium as gym
    >>> env = gym.make("Vampair-v0", max_clauses=10)
    >>> wrapped_env = UsefulActionsWrapper(env)

    although it's an action wrapper, the ``action`` method does nothing

    >>> wrapped_env.action(0)
    0
    >>> _ = wrapped_env.reset()
    >>> _, reward, _, _, _ = wrapped_env.step((0, 2))
    >>> reward
    0.0

    repeating an action is useless and results in a negative reward

    >>> _, reward, _, _, _ = wrapped_env.step((0, 2))
    >>> reward
    -1.0
    """

    def step(
        self, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Run the ``step`` and modify ``action``.

        :param action: action selected by an agent
        :return: standard Gymnasium ``step`` results
        """
        old_state_size = len(self.env.unwrapped.state.clauses)  # type: ignore
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        return (
            observation,
            -1.0
            if len(self.env.unwrapped.state.clauses)  # type: ignore
            == old_state_size
            else reward,
            terminated,
            truncated,
            info,
        )

    def action(self, action: Any) -> Any:
        """Return the non-modified action. Left for compatibility.

        :param action: the original  actions
        :returns: the non-modified actions
        """
        return action
