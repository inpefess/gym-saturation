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
from typing import Optional, Tuple, Type


class RelayServer(TCPServer):
    r"""
    A TCP server to read and write data from another opened TCP socket.

    >>> from threading import Thread
    >>> import random
    >>> data_port = random.randint(10000, 2 ** 16 - 1)
    >>> control_port = random.randint(10000, 2 ** 16 - 1)
    >>> with RelayServer(
    ...     data_port, ("localhost", control_port), RelayTCPHandler
    ... ) as relay_server:
    ...     thread = Thread(target=relay_server.serve_forever)
    ...     thread.daemon = True
    ...     thread.start()
    ...     with socket.create_connection(
    ...         ("localhost", data_port)
    ...     ) as data_connection:
    ...         data_connection.sendall(b"test\x00")
    ...         with socket.create_connection(
    ...             ("localhost", control_port)
    ...         ) as control_connection:
    ...             control_connection.sendall(b"READ")
    ...             print(control_connection.recv(4096))
    ...         with socket.create_connection(
    ...             ("localhost", control_port)
    ...         ) as control_connection:
    ...             control_connection.sendall(b"anything but READ")
    ...             print(control_connection.recv(4096).decode("utf8"))
    ...     relay_server.shutdown()
    ...     thread.join()
    b'test\x00'
    OK
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
        self.data_socket = socket.create_server(
            address=("localhost", data_port), family=socket.AF_INET
        )
        self._data_connection: Optional[socket.socket] = None

    @property
    def data_connection(self) -> socket.socket:
        """Get the connection to read from and write to."""
        if self._data_connection is None:
            self._data_connection = self.data_socket.accept()[0]
        return self._data_connection


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
