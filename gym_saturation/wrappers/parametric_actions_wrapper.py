#   Copyright 2023 Boris Shminke
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# noqa: D205, D400
"""
Parametric Actions Wrapper
===========================
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from gym_saturation.envs.saturation_env import (
    ACTION_MASK,
    REAL_OBS,
    SaturationEnv,
)

PARAMETRIC_ACTIONS = "avail_actions"


class ParamtericActionsWrapper(gym.Wrapper, ABC):
    """
    A parametric actions wrapper.

    .. _parametric_actions:

    It's incremental, i. e. it embeds only the new clauses in the observation.
    It defines the clauses as old if their order numbers are small than the
    previous step maximum.

    >>> tptp_folder = getfixture("mock_tptp_folder")  # noqa: F821
    >>> import os
    >>> problem_list = [os.path.join(tptp_folder, "Problems", "TST",
    ...     "TST003-1.p")]
    >>> class ConstantClauseWeight(ParamtericActionsWrapper):
    ...     def clause_embedder(self, clause: Dict[str, Any]) -> np.ndarray:
    ...         return np.ones(
    ...             (self.observation_space[PARAMETRIC_ACTIONS].shape[1],)
    ...         )
    >>> env = gym.make("Vampire-v0", problem_list=problem_list, max_clauses=5)
    >>> wrapped_env = ConstantClauseWeight(env, embedding_dim=1)
    >>> observation, info = wrapped_env.reset()
    >>> observation.keys()
    dict_keys(['action_mask', 'avail_actions'])
    >>> info.keys()
    dict_keys(['problem_filename', 'real_obs'])
    >>> observation[PARAMETRIC_ACTIONS]
    array([[1.],
           [1.],
           [1.],
           [0.],
           [0.]])
    >>> _ = wrapped_env.step(0)
    >>> observation, _, _, _, _ = wrapped_env.step(1)
    >>> observation[PARAMETRIC_ACTIONS]
    array([[1.],
           [1.],
           [1.],
           [1.],
           [1.]])
    """

    def __init__(
        self,
        env: SaturationEnv,
        embedding_dim: int,
    ):
        """Initialise all the things."""
        super().__init__(env)
        self.env: SaturationEnv = env  # type: ignore
        action_mask = self.env.observation_space[ACTION_MASK]
        parametric_actions = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                action_mask.shape[0],  # type: ignore
                embedding_dim,
            ),
        )
        self.observation_space = gym.spaces.Dict(
            {
                ACTION_MASK: env.observation_space[ACTION_MASK],
                PARAMETRIC_ACTIONS: parametric_actions,
            }
        )
        self.clause_embeddings = np.zeros(parametric_actions.shape)
        self.embedded_clauses_cnt = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment.

        :param seed: seed for compatibility
        :param options: options for compatibility
        """
        observation, info = self.env.reset(seed=seed, options=options)
        info["real_obs"] = observation
        self.embedded_clauses_cnt = 0
        return self.observation(observation), info

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a modified observation.

        :param observation: the unwrapped observation
        :returns: the modified observation
        """
        new_clauses = [
            clause["literals"]
            for clause in observation[REAL_OBS][self.embedded_clauses_cnt :]
        ]
        new_clauses_count = len(new_clauses)
        if (
            new_clauses_count > 0
            and self.embedded_clauses_cnt + new_clauses_count
            <= self.clause_embeddings.shape[0]
        ):
            self.clause_embeddings[
                self.embedded_clauses_cnt : self.embedded_clauses_cnt
                + new_clauses_count,
                :,
            ] = np.array(list(map(self.clause_embedder, new_clauses)))
            self.embedded_clauses_cnt += new_clauses_count
        return {
            ACTION_MASK: observation[ACTION_MASK],
            PARAMETRIC_ACTIONS: self.clause_embeddings,
        }

    def step(
        self, action: np.int64
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Make the environment step.

        :param action: agent's action of choice
        """
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        info[REAL_OBS] = observation
        return (
            self.observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    @abstractmethod
    def clause_embedder(self, literals: str) -> np.ndarray:
        """
        Embed the clause.

        :param literals: a TPTP clause literals to embed
        :returns: an embedding vector
        """
        raise NotImplementedError  # pragma: no cover
