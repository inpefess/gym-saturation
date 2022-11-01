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
Saturation Environment with iProver back-end
============================================
"""
import asyncio
import os
import random
import re
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import orjson

from gym_saturation.envs.saturation_env import MAX_CLAUSES, SaturationEnv
from gym_saturation.utils import FALSEHOOD_SYMBOL, Clause


async def _iprover_start(
    iprover_port: int, problem_filename: str, iprover_binary_path: str
) -> asyncio.subprocess.Process:
    tptp_folder = os.path.join(os.path.dirname(problem_filename), "..", "..")
    arguments = [
        "--interactive_mode",
        "true",
        "--enigma_ip_address",
        "127.0.0.1",
        "--enigma_port",
        str(iprover_port),
        "--schedule",
        "none",
        "--resolution_flag",
        "false",
        "--instantiation_flag",
        "false",
        "--superposition_flag",
        "true",
        "--sup_iter_deepening",
        "0",
        "--sup_passive_queues",
        "[[+enigma_score]]",
        "--sup_passive_queues_freq",
        "[1]",
        "--include_path",
        tptp_folder,
        problem_filename,
    ]
    return await asyncio.create_subprocess_exec(
        iprover_binary_path, *arguments
    )


async def _await_start_relay_server(iprover_port: int, agent_port: int):
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "from gym_saturation.relay_server import start_relay_server; "
        + f"start_relay_server({iprover_port}, {agent_port})",
    )


def _parse_batch_clauses(
    batch_clauses: List[Dict[str, Any]]
) -> Dict[str, Clause]:
    updated: Dict[str, Clause] = {}
    for dict_clause in batch_clauses:
        label, literals, inference_rule = re.findall(
            r"cnf\((\w+),\w+,([^,]*),(.*)\)\.",
            dict_clause["clause"].replace("\n", "").replace(" ", ""),
        )[0]
        changed_clause = Clause(
            literals=literals,
            label=label,
            birth_step=dict_clause["clause_features"]["born"] - 1,
            inference_rule=inference_rule,
        )
        updated[label] = changed_clause
    return updated


def _parse_szs_status(szs_status: str) -> Dict[str, Clause]:
    status_code = re.findall(
        r"\% SZS status (\w+) for .+\.p",
        szs_status.replace("\n", ""),
    )[0]
    if status_code == "Unsatisfiable":
        dummy_clause = Clause(
            literals=FALSEHOOD_SYMBOL, inference_rule="dummy"
        )
        return {dummy_clause.label: dummy_clause}
    raise ValueError(f"unexpected status: {status_code}")


def _parse_iprover_response(iprover_response: bytes) -> Dict[str, Clause]:
    json_payload = orjson.loads(
        iprover_response.decode("utf8").replace("\x00", "").replace("\n", "")
    )
    if "batch_clauses" in json_payload:
        return _parse_batch_clauses(json_payload["batch_clauses"])
    return _parse_szs_status(json_payload["szs_status"])


class IProverEnv(SaturationEnv):
    """An RL environment around iProver."""

    def __init__(
        self,
        iprover_binary_path: str,
        port_pair: Tuple[int, int],
        problem_list: List[str],
        max_clauses: int = MAX_CLAUSES,
    ):
        """
        Initialise the environment.

        :param iprover_binary_path: a path to iProver binary
        :param port_pair: iProver will connect to the first port,
            a port to listen for agent's connection is the second one
        :param problem_list: a list of names of TPTP problem files
        :param max_clauses: maximal number of clauses in proof state
        """
        super().__init__(problem_list, max_clauses)
        self.iprover_port, self.agent_port = port_pair
        self.iprover_binary_path = iprover_binary_path
        self._scores_number = 0
        self._tcp_socket: Optional[socket.socket] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[Dict, Tuple[dict, dict]]:  # noqa: D102
        self.problem = random.choice(self.problem_list)
        asyncio.run(
            _await_start_relay_server(self.iprover_port, self.agent_port)
        )
        time.sleep(1)
        asyncio.run(
            _iprover_start(
                self.iprover_port, self.problem, self.iprover_binary_path
            )
        )
        time.sleep(1)
        self._tcp_socket = socket.create_connection(
            ("localhost", self.agent_port)
        )
        data = self._get_data()
        self._state = _parse_iprover_response(data)
        self._scores_number = len(self._state)
        return self.state

    @property
    def tcp_socket(self) -> socket.socket:
        """
        Get a TCP socket for communicating actions.

        :raises ValueError: is the TCP socket is not initialised
        """
        if self._tcp_socket:
            return self._tcp_socket
        raise ValueError("open tcp_socket before using it!")

    def _get_data(self) -> bytes:
        data = b"\x00\n"
        while data == b"\x00\n":
            while True:
                raw_data = self.tcp_socket.recv(4096)
                if raw_data:
                    data += raw_data
                else:
                    data = raw_data
                if (
                    not raw_data
                    or raw_data[-2:] == b"\x00\n"
                    or raw_data[-1] == b"\x00"
                ):
                    break
        return data

    def _do_deductions(self, action: int) -> Tuple[bytes, ...]:
        scores = (
            '{"scores":['
            + (self._scores_number * f"{1 / (action + 1)},")[:-1]
            + "]}\n\x00\n"
        )
        self.tcp_socket.sendall(scores.encode("utf8"))
        data = self._get_data()
        if data:
            updated = _parse_iprover_response(data)
        self._scores_number = len(updated)
        self._state.update(updated)
        return tuple(map(orjson.dumps, updated.values()))
