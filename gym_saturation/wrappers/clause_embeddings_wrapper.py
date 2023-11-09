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
Clause Embeddings Wrapper
==========================
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper

from gym_saturation.constants import CLAUSE_EMBEDDINGS
from gym_saturation.envs.saturation_env import SaturationEnv


class ClauseEmbeddingsWrapper(ObservationWrapper, ABC):
    """
    A clause embeddings wrapper.

    It's incremental, i.e. it embeds only the new clauses in the observation.
    It defines the clauses as old if their order numbers are smaller than the
    previous step maximum.

    >>> class ConstantClauseWeight(ClauseEmbeddingsWrapper):
    ...     def clause_embedder(self, clause: Dict[str, Any]) -> np.ndarray:
    ...         return np.ones(
    ...             (self.observation_space[CLAUSE_EMBEDDINGS].shape[1],)
    ...         )
    >>> env = gym.make("Vampire-v0", max_clauses=10)
    >>> wrapped_env = ConstantClauseWeight(env, embedding_dim=1)
    >>> observation, info = wrapped_env.reset()
    >>> observation.keys()
    dict_keys(['clause_embeddings'])
    >>> info.keys()
    dict_keys(['clauses'])
    >>> observation[CLAUSE_EMBEDDINGS]
    array([[1.],
           [1.],
           [1.],
           [1.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.]])
    >>> _ = wrapped_env.step(0)
    >>> observation, _, _, _, _ = wrapped_env.step(1)
    >>> observation[CLAUSE_EMBEDDINGS]
    array([[1.],
           [1.],
           [1.],
           [1.],
           [1.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.]])
    """

    def __init__(
        self,
        env: SaturationEnv,
        embedding_dim: int,
    ):
        """Initialise all the things."""
        super().__init__(env)
        clause_embeddings = gym.spaces.Box(
            low=-np.infty,
            high=np.infty,
            shape=(
                self.env.unwrapped.state.max_clauses,  # type: ignore
                embedding_dim,
            ),
        )
        self.observation_space = gym.spaces.Dict(
            {
                CLAUSE_EMBEDDINGS: clause_embeddings,
            }
        )
        self.clause_embeddings = np.zeros(clause_embeddings.shape)
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
        :returns: observations and info
        """
        observation, info = super().reset(seed=seed, options=options)
        info["clauses"] = observation
        self.embedded_clauses_cnt = 0
        return observation, info

    def observation(
        self, observation: Tuple[Dict[str, Any], ...]
    ) -> Dict[str, Any]:
        """
        Return a modified observation.

        :param observation: the unwrapped observation
        :returns: the modified observation
        """
        new_clauses = [
            clause["literals"]
            for clause in observation[self.embedded_clauses_cnt :]
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
            CLAUSE_EMBEDDINGS: self.clause_embeddings,
        }

    @abstractmethod
    def clause_embedder(self, literals: str) -> np.ndarray:
        """
        Embed the clause.

        :param literals: a TPTP clause literals to embed
        :returns: an embedding vector
        """
        raise NotImplementedError  # pragma: no cover
