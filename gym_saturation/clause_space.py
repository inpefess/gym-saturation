"""
Copyright 2021 Boris Shminke

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from gym import spaces

from gym_saturation.grammar import Clause
from gym_saturation.parsing.json_grammar import dict_to_clause


class ClauseSpace(spaces.Space):
    """
    an OpenAI Gym space for a list of clauses

    >>> space = ClauseSpace()
    >>> space.sample() in space
    True
    >>> "no" in space
    False
    >>> [{"class": "no"}] in space
    False
    """

    def __init__(self):
        super().__init__(shape=(0,))

    def contains(self, x):
        if isinstance(x, list):
            for clause in x:
                try:
                    return isinstance(dict_to_clause(clause), Clause)
                except (ValueError, KeyError):
                    return False
        return False

    def sample(self):
        return [
            {
                "class": "Clause",
                "literals": [],
                "label": "test",
                "birth_step": 0,
                "processed": False,
                "inference_parents": [],
                "inference_rule": "",
            }
        ]
