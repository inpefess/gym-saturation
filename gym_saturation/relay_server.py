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
    r"""
    A TCP server to read and write data from another opened TCP socket.

    >>> from threading import Thread
    >>> from time import sleep
    >>> data_port = 10000
    >>> control_port = 10001
    >>> thread = Thread(
    ...     target=start_relay_server,
    ...     args=(data_port, control_port)
    ... )
    >>> thread.daemon = True
    >>> thread.start()
    >>> sleep(1)
    >>> data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    >>> data_socket.connect(("localhost", data_port))
    >>> data_socket.sendall(b"test\x00")
    >>> def send_command(command: bytes) -> bytes:
    ...     control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ...     control_socket.connect(("localhost", control_port))
    ...     control_socket.sendall(command)
    ...     return control_socket.recv(4096)
    >>> print(send_command(b"anything but READ").decode("utf8"))
    OK
    >>> print(send_command(b"READ"))
    b'test\x00'
    """

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
        command = self.request.recv(4096)
        if command == b"READ":
            input_data = b""
            while b"\x00" not in input_data[-3:]:
                input_data += self.server.data_connection.recv(4096)
            self.request.sendall(input_data)
        else:
            self.server.data_connection.sendall(command)
            self.request.sendall(b"OK")


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
