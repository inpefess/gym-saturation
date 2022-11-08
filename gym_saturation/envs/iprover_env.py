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
import subprocess
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import orjson

from gym_saturation.envs.saturation_env import MAX_CLAUSES, SaturationEnv
from gym_saturation.relay_server import RelayServer, RelayTCPHandler
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
        "--interactive_mode",
        "true",
        "--sup_passive_queue_type",
        "external_agent",
        "--preprocessing_flag",
        "false",
        "--include_path",
        tptp_folder,
        problem_filename,
    ]
    return await asyncio.create_subprocess_exec(
        iprover_binary_path, *arguments, stdout=subprocess.DEVNULL
    )


def _parse_batch_clauses(
    batch_clauses: List[Dict[str, Any]]
) -> Dict[str, Clause]:
    updated: Dict[str, Clause] = {}
    for dict_clause in batch_clauses:
        raw_clause = dict_clause["clause"].replace("\n", "").replace(" ", "")
        try:
            label, literals, inference_rule, inference_parents = re.findall(
                r"cnf\((\w+),\w+,(.*),inference\((.*),.*,\[(.*)\]\)\)\.",
                raw_clause,
            )[0]
        except IndexError:
            label, literals = re.findall(
                r"cnf\((\w+),\w+,(.*),file\(.*\)\)\.",
                raw_clause,
            )[0]
            inference_rule, inference_parents = "input", None
        changed_clause = Clause(
            literals=literals,
            label=label,
            birth_step=dict_clause["clause_features"]["born"] - 1,
            inference_rule=inference_rule,
            inference_parents=tuple(inference_parents.split(","))
            if inference_parents is not None
            else None,
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


def _parse_iprover_requests(
    iprover_requests: List[Dict[str, Any]]
) -> Dict[str, Clause]:
    state_update = {}
    for iprover_request in iprover_requests:
        if "clauses" in iprover_request:
            state_update.update(
                _parse_batch_clauses(iprover_request["clauses"])
            )
        if "szs_status" in iprover_request:
            state_update.update(
                _parse_szs_status(iprover_request["szs_status"])
            )
    return state_update


class IProverEnv(SaturationEnv):
    """
    An RL environment around iProver.

    >>> import sys
    >>> if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> from glob import glob
    >>> problems = sorted(glob(os.path.join(files("gym_saturation").joinpath(
    ...     os.path.join("resources", "TPTP-mock", "Problems")
    ... ), "SET", "*-*.p")))
    >>> iprover_env = IProverEnv(problems)
    >>> observation = iprover_env.reset()
    >>> for action in [0, 1, 2, 4, 8, 9, 10]:
    ...     observation, reward, done, info = iprover_env.step(action)
    >>> print(reward, done)
    1.0 True
    >>> iprover_env.close()
    """

    def __init__(
        self,
        problem_list: List[str],
        max_clauses: int = MAX_CLAUSES,
        port_pair: Optional[Tuple[int, int]] = None,
        iprover_binary_path: str = "iproveropt",
    ):
        """
        Initialise the environment.

        :param problem_list: a list of names of TPTP problem files
        :param max_clauses: maximal number of clauses in proof state
        :param port_pair: iProver will connect to the first port,
            a port to listen for agent's connection is the second one
        :param iprover_binary_path: a path to iProver binary;
            by default, we assume it to be ``iproveropt`` and in the $PATH
        """
        super().__init__(problem_list, max_clauses)
        (
            self.iprover_port,
            self.agent_port,
        ) = (  # pylint: disable=unbalanced-tuple-unpacking
            (
                random.randint(10000, 2**16 - 1),
                random.randint(10000, 2**16 - 1),
            )
            if port_pair is None
            else port_pair
        )
        self.iprover_binary_path = iprover_binary_path
        self.relay_server = RelayServer(
            self.iprover_port, ("localhost", self.agent_port), RelayTCPHandler
        )
        self.relay_server_thread = Thread(
            target=self.relay_server.serve_forever
        )
        self.relay_server_thread.daemon = True
        self.relay_server_thread.start()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[Dict, Tuple[dict, dict]]:  # noqa: D102
        self.problem = random.choice(self.problem_list)
        asyncio.run(
            _iprover_start(
                self.iprover_port, self.problem, self.iprover_binary_path
            )
        )
        time.sleep(2)
        data = self._get_json_data()
        self._state = _parse_iprover_requests(data)
        return self.state

    def _get_raw_data(self) -> bytes:
        with socket.create_connection(
            ("localhost", self.agent_port)
        ) as tcp_socket:
            tcp_socket.sendall(b"READ")
            raw_data = tcp_socket.recv(4096)
            while b"\x00" not in raw_data[-3:]:
                raw_data += tcp_socket.recv(4096)
        return raw_data

    def _get_json_data(self) -> List[Dict[str, Any]]:
        json_data = [{"query": "None"}]
        while json_data[-1]["query"] not in {
            "given_clause_request",
            "szs_status",
            "proof_out",
        }:
            raw_data = self._get_raw_data()
            raw_jsons = [
                request.replace("\n", "")
                for request in raw_data.decode("utf8").split("\x00")
            ]
            for raw_json in raw_jsons:
                if raw_json:
                    parsed_json = orjson.loads(
                        raw_json.replace("\n", "").replace("\x00", "")
                    )
                    json_data.append(parsed_json)
        return json_data[1:]

    def _do_deductions(self, action: int) -> Tuple[bytes, ...]:
        given_clause_label = list(self._state.values())[action].label[2:]
        scores = (
            f"""{{"given_clause": {given_clause_label},"""
            + """"passive_is_empty": false}\n\x00\n"""
        )
        with socket.create_connection(
            ("localhost", self.agent_port)
        ) as tcp_socket:
            tcp_socket.sendall(scores.encode("utf8"))
        data = self._get_json_data()
        if data:
            updated = _parse_iprover_requests(data)
        self._state.update(updated)
        return tuple(map(orjson.dumps, updated.values()))

    def close(self) -> None:
        """Stop relay server."""
        self.relay_server.data_connection.close()
        self.relay_server.shutdown()
        self.relay_server.server_close()
        self.relay_server_thread.join()
