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


def _send_data_from_one_connection_to_another(
    source_connection: socket.socket, target_connection: socket.socket
) -> bytes:
    while True:
        data = source_connection.recv(4096)
        if data:
            target_connection.sendall(data)
        if not data or data[-2:] == b"\x00\n" or data[-1] == b"\x00":
            break
    return data


def _bind_and_listen_to_two_ports(
    socket_one: socket.socket,
    port_one: int,
    socket_two: socket.socket,
    port_two: int,
) -> None:
    socket_one.bind(("localhost", port_one))
    socket_one.listen()
    socket_two.bind(("localhost", port_two))
    socket_two.listen()


def start_relay_server(port_one: int, port_two: int) -> None:
    """
    Start a server sending data as is from one port to another and vice versa.

    :param port_one: port from which we read first
    :param port_two: port to which we write first
    """
    with socket.socket(
        socket.AF_INET, socket.SOCK_STREAM
    ) as socket_one, socket.socket(
        socket.AF_INET, socket.SOCK_STREAM
    ) as socket_two:
        _bind_and_listen_to_two_ports(
            socket_one, port_one, socket_two, port_two
        )
        with socket_one.accept()[0] as connection_one, socket_two.accept()[
            0
        ] as connection_two:
            while True:
                data_one = _send_data_from_one_connection_to_another(
                    connection_one, connection_two
                )
                if not data_one:
                    break
                data_two = _send_data_from_one_connection_to_another(
                    connection_two, connection_one
                )
                if not data_two:
                    break
