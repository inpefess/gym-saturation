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
ast2vec Wrapper
================
"""
import json
from urllib.request import HTTPHandler, OpenerDirector, Request

import numpy as np

from gym_saturation.utils import tptp2python
from gym_saturation.wrappers.clause_embeddings_wrapper import (
    ClauseEmbeddingsWrapper,
)


class AST2VecWrapper(ClauseEmbeddingsWrapper):
    """
    An ast2vec wrappers for saturation provers.

    .. _ast2vec_wrapper:

    The best way is to run TorchServe docker container as described here:
    https://gitlab.com/inpefess/ast2vec

    >>> import gymnasium as gym
    >>> env = gym.make("Vampire-v0", max_clauses=9)
    >>> wrapped_env = AST2VecWrapper(env, features_num=256)
    >>> observation, info = wrapped_env.reset()
    >>> observation.keys()
    dict_keys(['clause_embeddings'])
    >>> from gym_saturation.wrappers.clause_embeddings_wrapper import (
    ...     CLAUSE_EMBEDDINGS)
    >>> observation[CLAUSE_EMBEDDINGS].shape
    (9, 256)
    """

    def __init__(
        self,
        env,
        features_num: int,
        torch_serve_url: str = "http://127.0.0.1:9080/predictions/ast2vec",
    ):
        """Initialise all the things."""
        super().__init__(env, features_num)
        self.torch_serve_url = torch_serve_url

    def clause_embedder(self, literals: str) -> np.ndarray:
        """
        Embed the clause.

        :param literals: a TPTP clause literals to embed
        :returns: an embedding vector
        """
        prepared_literals = tptp2python(literals)
        req = Request(
            self.torch_serve_url,
            f'{{"data": "{prepared_literals}"}}'.encode(),
            {"Content-Type": "application/json"},
        )
        opener = OpenerDirector()
        opener.add_handler(HTTPHandler())
        with opener.open(req) as response:
            clause_embedding = json.loads(response.read().decode("utf-8"))
        return np.array(clause_embedding)
