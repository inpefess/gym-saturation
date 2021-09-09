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
import pickle
from typing import Any, List


def deduplicate(a_list: List[Any]) -> List[Any]:
    """
    deduplicate a list

    :param a_list: a list of possibly repeating items
    :returns: a list of unique items
    """
    new_list = []
    for item in a_list:
        if item not in new_list:
            new_list.append(item)
    return new_list


def pickle_copy(obj: object):
    """ faster ``deepcopy`` analogue with ``pickle`` """
    return pickle.loads(pickle.dumps(obj))
