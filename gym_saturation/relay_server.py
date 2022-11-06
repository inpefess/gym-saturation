# Copyright 2022 Boris Shminke
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
Relay Server between Two Sockets
================================
"""
import socket
from socketserver import BaseRequestHandler, TCPServer
from typing import Tuple, Type


class RelayServer(TCPServer):
    """A TCP server to read and write data from another opened TCP socket."""

    def __init__(
        self,
        data_port: int,
        server_address: Tuple[str, int],
        request_handler_class: Type[BaseRequestHandler],
        bind_and_activate: bool = True,
    ):
        """Accept a data connection."""
        super().__init__(
            server_address, request_handler_class, bind_and_activate
        )
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_socket.bind(("localhost", data_port))
        data_socket.listen()
        self.data_connection = data_socket.accept()[0]


class RelayTCPHandler(BaseRequestHandler):
    """The request handler class for relay server."""

    def handle(self):
        """Read data from another TCP socket or send data to it."""
        data = self.request.recv(4096)
        if data == b"READ":
            input_data = self.server.data_connection.recv(4096)
            while b"\x00" not in input_data[-3:]:
                input_data += self.server.data_connection.recv(4096)
            self.request.sendall(input_data)
        else:
            self.server.data_connection.sendall(data)
            self.request.sendall(b"OK\n")


def start_relay_server(data_port: int, control_port: int) -> None:
    """
    Start a relay server.

    :param data_port: a port to read and write data (no commands)
    :param control_port: a port to listen for commands
    """
    with RelayServer(
        data_port, ("localhost", control_port), RelayTCPHandler
    ) as server:
        server.serve_forever()
