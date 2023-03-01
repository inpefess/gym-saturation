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
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Generator

from pytest import fixture

if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    # pylint: disable=no-name-in-module
    from importlib.abc import Traversable  # type: ignore

    # pylint: disable=no-name-in-module
    from importlib.resources import files  # type: ignore
else:  # pragma: no cover
    from pathlib import PosixPath as Traversable  # noqa: F401

    from importlib_resources import files  # pylint: disable=import-error


@fixture
def mock_tptp_folder() -> Traversable:
    """Return the mock TPTP folder path."""
    return files("gym_saturation").joinpath(
        os.path.join("resources", "TPTP-mock")
    )


class DummyHTTPHandler(BaseHTTPRequestHandler):
    """A dummy handler transforming strings to vectors."""

    # pylint: disable=invalid-name
    def do_POST(self) -> None:
        """Respond with 256 float ones as a dummy embedding."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(str(256 * [1.0]).encode("utf-8"))


@fixture(autouse=True, scope="session")
def http_server() -> Generator[HTTPServer, None, None]:
    """
    Mock TorchServe behaviour with a simplistic HTTP server.

    :returns: a mock HTTP server
    """
    with HTTPServer(("localhost", 9080), DummyHTTPHandler) as server:
        thread = Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        yield server
