# Copyright 2021-2022 Boris Shminke

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# noqa: D205, D400
"""
Clause Space
=============
"""
import orjson
from gym import spaces


class ClauseSpace(spaces.Space):
    """
    An OpenAI Gym space for a list of bytes.

    .. _clause_space:

    >>> space = ClauseSpace()
    >>> space.sample() in space
    True
    >>> "no" in space
    False
    >>> ["no"] in space
    False
    """

    def __init__(self):  # noqa: D107
        super().__init__(shape=(0,))

    def contains(self, x):  # noqa: D102
        if isinstance(x, list):
            for byte_sequence in x:
                return isinstance(byte_sequence, bytes)
        return False

    def sample(self):  # noqa: D102
        return [
            orjson.dumps(
                {
                    "literals": (),
                    "label": "test",
                    "role": "lemma",
                    "birth_step": 0,
                    "processed": False,
                    "inference_parents": (),
                    "inference_rule": "",
                }
            )
        ]
