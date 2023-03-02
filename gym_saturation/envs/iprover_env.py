# Copyright 2022-2023 Boris Shminke
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
import json
import os
import random
import re
import socket
import subprocess
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gym_saturation.envs.saturation_env import (
    ACTION_MASK,
    MAX_CLAUSES,
    PROBLEM_FILENAME,
    REAL_OBS,
    SaturationEnv,
)
from gym_saturation.relay_server import RelayServer, RelayTCPHandler
from gym_saturation.utils import FALSEHOOD_SYMBOL


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


class IProverEnv(SaturationEnv):
    """
    An RL environment around iProver.

    >>> tptp_folder = getfixture("mock_tptp_folder")  # noqa: F821
    >>> from glob import glob
    >>> problems = sorted(glob(os.path.join(tptp_folder, "Problems", "SET",
    ...     "*-*.p")))
    >>> iprover_env = IProverEnv(problems)
    >>> observation, info = iprover_env.reset()
    >>> for action in [0, 1, 2, 4, 8, 9, 10]:
    ...     observation, reward, terminated, truncated, info = (iprover_env.
    ...         step(action))
    >>> print(reward, terminated, truncated)
    1.0 True False
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

    def _parse_batch_clauses(
        self, batch_clauses: List[Dict[str, Any]]
    ) -> None:
        for dict_clause in batch_clauses:
            raw_clause = (
                dict_clause["clause"].replace("\n", "").replace(" ", "")
            )
            try:
                (
                    label,
                    literals,
                    inference_rule,
                    inference_parents,
                ) = re.findall(
                    r"cnf\((\w+),\w+,(.*),inference\((.*),.*,\[(.*)\]\)\)\.",
                    raw_clause,
                )[
                    0
                ]
            except IndexError:
                label, literals = re.findall(
                    r"cnf\((\w+),\w+,(.*),file\(.*\)\)\.",
                    raw_clause,
                )[0]
                inference_rule, inference_parents = "input", None
            self.state.add_clause(
                {
                    "literals": literals,
                    "label": label,
                    "role": "lemma",
                    "birth_step": dict_clause["clause_features"]["born"] - 1,
                    "inference_rule": inference_rule,
                    "inference_parents": tuple(inference_parents.split(","))
                    if inference_parents is not None
                    else (),
                }
            )
            self.state.set_action_mask_by_label(label, 1.0)

    def _parse_szs_status(self, szs_status: str) -> None:
        status_code = re.findall(
            r"\% SZS status (\w+) for .+\.p",
            szs_status.replace("\n", ""),
        )[0]
        if status_code == "Unsatisfiable":
            self.state.add_clause(
                {
                    "label": "dummy",
                    "literals": FALSEHOOD_SYMBOL,
                    "inference_rule": "dummy",
                    "inference_parents": (),
                    "role": "lemma",
                }
            )
        else:
            raise ValueError(f"unexpected status: {status_code}")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # noqa: D102
        super().reset(seed=seed)
        asyncio.run(
            _iprover_start(
                self.iprover_port, self.get_task()[0], self.iprover_binary_path
            )
        )
        time.sleep(2)
        data = self._get_json_data()
        self._parse_iprover_requests(data)
        return {
            REAL_OBS: self.state.clauses,
            ACTION_MASK: self.state.action_mask,
        }, {PROBLEM_FILENAME: self.problem_filename}

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
                    parsed_json = json.loads(
                        raw_json.replace("\n", "").replace("\x00", "")
                    )
                    json_data.append(parsed_json)
        return json_data[1:]

    def _parse_iprover_requests(
        self, iprover_requests: List[Dict[str, Any]]
    ) -> None:
        for iprover_request in iprover_requests:
            if "clauses" in iprover_request:
                self._parse_batch_clauses(iprover_request["clauses"])
            if "szs_status" in iprover_request:
                self._parse_szs_status(iprover_request["szs_status"])

    def _do_deductions(self, action: np.int64) -> None:
        given_clause_label = self.state.clause_labels[action][2:]
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
            self._parse_iprover_requests(data)

    def close(self) -> None:
        """Stop relay server."""
        self.relay_server.data_connection.close()
        self.relay_server.shutdown()
        self.relay_server.server_close()
        self.relay_server_thread.join()
