# Copyright 2023 Boris Shminke
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
Constants used throughout the package
======================================
"""
import os
import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    # pylint: disable=no-name-in-module
    from importlib.resources import files  # type: ignore  # pragma: no cover
else:  # pragma: no cover
    from importlib_resources import files  # pylint: disable=import-error

CLAUSE_EMBEDDINGS = "clause_embeddings"
FALSEHOOD_SYMBOL = "$false"
MOCK_TPTP_FOLDER = str(
    files("gym_saturation").joinpath(os.path.join("resources", "TPTP-mock"))
)
MOCK_TPTP_PROBLEM = os.path.join(
    MOCK_TPTP_FOLDER, "Problems", "TST", "TST001-1.p"
)
