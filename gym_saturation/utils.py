# Copyright 2021-2023 Boris Shminke
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
Logic Operations Utility Functions
===================================
"""
from http.server import HTTPServer
from threading import Thread
from typing import Any, Dict


def pretty_print(clause: Dict[str, Any]) -> str:
    """
    Print a logical formula back to TPTP language.

    :param clause: a logical clause to print
    :returns: a TPTP string
    """
    res = f"cnf({clause['label']}, {clause['role']}, "
    res += clause["literals"]
    res += (
        f", inference({clause['inference_rule']}, [], ["
        + ", ".join(clause["inference_parents"])
        + "])"
    )
    return res + ")."


def tptp2python(literals: str) -> str:
    """
    Transform a TPTP CNF formula literals to look like Python code.

    :param literals: literals of a TPTP-formatted first-order formula in CNF
    :return: (hopefully syntactically correct) Python Boolean-valued expression
    """
    return (
        literals.replace("==", "^^")
        .replace("!=", "^^^")
        .replace("=", "==")
        .replace("^^^", "!=")
        .replace("^^", "==")
        .replace("$false", "False")
        .replace("as", "__as")
    )


def start_server_in_a_thread(server: HTTPServer) -> None:
    """
    Start a given HTTP server as a daemon.

    :param server: HTTP server
    """
    thread = Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
