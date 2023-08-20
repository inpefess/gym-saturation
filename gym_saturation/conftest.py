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
Fixtures for the Tests
=======================
"""
from http.server import HTTPServer
from typing import Generator

from pytest import fixture

from gym_saturation.dummy_http_handler import DummyHTTPHandler
from gym_saturation.utils import start_server_in_a_thread


@fixture(autouse=True, scope="session")
def http_server() -> Generator[HTTPServer, None, None]:
    """
    Mock TorchServe behaviour with a simplistic HTTP server.

    :returns: a mock HTTP server
    """
    with HTTPServer(("localhost", 9080), DummyHTTPHandler) as server:
        start_server_in_a_thread(server)
        yield server


class DummyCodeBERTHandler(DummyHTTPHandler):
    """Dummy handler with a different embedding dimension."""

    _num_features = 768


@fixture(autouse=True, scope="session")
def codebert_features_server() -> Generator[HTTPServer, None, None]:
    """
    Mock a CodeBERT features server behaviour with a simplistic HTTP server.

    :returns: a mock HTTP server
    """
    with HTTPServer(("localhost", 7860), DummyCodeBERTHandler) as server:
        start_server_in_a_thread(server)
        yield server
