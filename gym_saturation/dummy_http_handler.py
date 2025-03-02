# Copyright 2023-2025 Boris Shminke
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
Dummy HTTP Handler for tests
=============================
"""
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler


class DummyHTTPHandler(BaseHTTPRequestHandler):
    """A dummy handler transforming strings to vectors."""

    _num_features = 256

    # pylint: disable=invalid-name
    def do_POST(self) -> None:
        """Respond with ``self._num_features`` ones as a dummy embedding."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(str(self._num_features * [1.0]).encode("utf-8"))

    # pylint: disable=invalid-name
    def do_GET(self) -> None:
        """Respond with ``self._num_features`` ones as a dummy embedding."""
        self.send_response(HTTPStatus.OK)
        self.end_headers()
        self.wfile.write(str(self._num_features * [1.0]).encode("utf-8"))
