# Copyright 2022-2025 Boris Shminke
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

"""
Relay Server between Two Sockets
================================
"""  # noqa: D205, D400

import json
from queue import Queue
from socketserver import BaseRequestHandler, ThreadingTCPServer
from typing import Any

QUERY_END_MESSAGE = "server_queries_start"
SESSION_END_MESSAGE = "proof_out"


class RelayServer(ThreadingTCPServer):
    r"""
    Server relaying i/o of a long-running connection to a TCP socket to queues.

    >>> import socket
    >>> from threading import Thread
    >>> with RelayServer(("localhost", 0), RelayTCPHandler
    ... ) as relay_server:
    ...     thread = Thread(target=relay_server.serve_forever)
    ...     thread.daemon = True
    ...     thread.start()
    ...     with socket.create_connection(relay_server.server_address
    ...     ) as socket_connection:
    ...         # data sent to the socket are stored in one queue
    ...         socket_connection.sendall(bytes(
    ...             f'{{"tag": "{QUERY_END_MESSAGE}"}}\n\x00\n', "utf8")
    ...         )
    ...         print(relay_server.input_queue.get())
    ...         # data from another queue are sent to the client
    ...         relay_server.output_queue.put(b"test")
    ...         print(str(socket_connection.recv(4096), "utf8"))
    ...         # message format for closing connection
    ...         socket_connection.sendall(bytes(
    ...             f'{{"tag": "{SESSION_END_MESSAGE}"}}\n\x00\n', "utf8"
    ...         ))
    ...     relay_server.shutdown()
    ...     thread.join()
    {'tag': 'server_queries_start'}
    test
    """

    def __init__(  # noqa: D107
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseRequestHandler],
        bind_and_activate: bool = True,
    ):
        super().__init__(
            server_address, request_handler_class, bind_and_activate
        )
        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        self.daemon_threads = True


class RelayTCPHandler(BaseRequestHandler):
    """The request handler class for relay server."""

    def read_messages(self) -> list[dict[str, Any]]:
        """
        Read messages from TCP request.

        :returns: parsed messages
        """
        raw_data = ""
        json_messages: list[dict[str, Any]] = []
        while len(json_messages) == 0 or json_messages[-1]["tag"] not in {
            QUERY_END_MESSAGE,
            SESSION_END_MESSAGE,
        }:
            raw_data += str(self.request.recv(4096), "utf8")
            raw_jsons = raw_data.split("\n\x00\n")
            raw_data = raw_jsons[-1]
            json_messages.extend(list(map(json.loads, raw_jsons[:-1])))
        return json_messages

    def handle(self) -> None:
        """Read data from another TCP socket or send data to it."""
        if isinstance(self.server, RelayServer):
            json_messages: list[dict[str, Any]] = []
            while (
                len(json_messages) == 0
                or json_messages[-1]["tag"] != SESSION_END_MESSAGE
            ):
                json_messages = self.read_messages()
                for json_message in json_messages:
                    self.server.input_queue.put(json_message)
                if json_messages[-1]["tag"] == QUERY_END_MESSAGE:
                    self.request.sendall(self.server.output_queue.get())
                    self.server.output_queue.task_done()
