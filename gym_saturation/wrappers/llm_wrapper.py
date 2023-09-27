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
Large Language Model Wrapper
=============================
"""
import json
from urllib.parse import urlencode
from urllib.request import HTTPHandler, build_opener

import numpy as np

from gym_saturation.utils import tptp2python
from gym_saturation.wrappers.clause_embeddings_wrapper import (
    ClauseEmbeddingsWrapper,
)


class LLMWrapper(ClauseEmbeddingsWrapper):
    """
    A Large Language Model wrapper for saturation provers.

    An example Docker image: https://gitlab.com/inpefess/codebert-features

    >>> import gymnasium as gym
    >>> env = gym.make("Vampire-v0", max_clauses=9)
    >>> wrapped_env = LLMWrapper(env, features_num=768)
    >>> observation, info = wrapped_env.reset()
    >>> observation.keys()
    dict_keys(['clause_embeddings'])
    >>> from gym_saturation.wrappers.clause_embeddings_wrapper import (
    ...     CLAUSE_EMBEDDINGS)
    >>> observation[CLAUSE_EMBEDDINGS].shape
    (9, 768)
    """

    def __init__(
        self,
        env,
        features_num: int,
        api_url: str = "http://127.0.0.1:7860/",
    ):
        """Initialise all the things."""
        super().__init__(env, features_num)
        self.api_url = api_url

    def clause_embedder(self, literals: str) -> np.ndarray:
        """
        Embed the clause.

        :param literals: a TPTP clause literals to embed
        :returns: an embedding vector
        """
        data = {"code_snippet": tptp2python(literals)}
        with build_opener(HTTPHandler()).open(
            f"{self.api_url}?{urlencode(data)}"
        ) as response:
            clause_embedding = response.read().decode("utf8")
        return np.array(json.loads(clause_embedding))
