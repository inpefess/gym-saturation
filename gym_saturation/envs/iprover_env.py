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
import re
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium.spaces import Discrete

from gym_saturation.envs.saturation_env import MAX_CLAUSES, SaturationEnv
from gym_saturation.relay_server import (
    QUERY_END_MESSAGE,
    SESSION_END_MESSAGE,
    RelayServer,
    RelayTCPHandler,
)


def _iprover_start(
    iprover_port: int, problem_filename: str, prover_binary_path: str
) -> asyncio.subprocess.Process:
    tptp_folder = os.path.join(os.path.dirname(problem_filename), "..", "..")
    command = " ".join(
        [
            prover_binary_path,
            "--interactive_mode",
            "true",
            "--external_ip_address",
            "127.0.0.1",
            "--external_port",
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
            "--sup_passive_queue_type",
            "external_agent",
            "--preprocessing_flag",
            "false",
            "--include_path",
            tptp_folder,
            problem_filename,
        ]
    )
    return asyncio.run(
        asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    )


class IProverEnv(SaturationEnv):
    """
    An RL environment around iProver.

    Refer to :ref:`saturation_env` for more documentation.

    >>> from gymnasium.utils.env_checker import check_env
    >>> import gymnasium as gym
    >>> env = gym.make(
    ...     "iProver-v0",
    ...     max_clauses=5
    ... ).unwrapped
    >>> env.relay_server
    Traceback (most recent call last):
     ...
    ValueError: run ``reset`` first!
    >>> check_env(env)
    cnf(c_53, ...).
    ...
    cnf(c_49, ...).

    when episode truncated, all threads are terminated

    >>> _ = env.reset()
    >>> _, _, _, truncated, _ = env.step(4)
    >>> truncated
    True
    """

    def __init__(
        self,
        max_clauses: int = MAX_CLAUSES,
        render_mode: str = "human",
        prover_binary_path: str = "iproveropt",
    ):
        """
        Initialise the environment.

        :param max_clauses: maximal number of clauses in proof state
        :param render_mode: a mode of running ``render`` method
        :param prover_binary_path: a path to iProver binary;
            by default, we assume it to be ``iproveropt`` and in the $PATH
        """
        super().__init__(max_clauses, render_mode)
        self.action_space = Discrete(self.state.max_clauses)
        self.prover_binary_path = prover_binary_path
        self._relay_server: Optional[RelayServer] = None
        self.relay_server_thread: Optional[Thread] = None
        self.iprover_process: Optional[asyncio.subprocess.Process] = None

    def _restart_relay_server(self) -> None:
        if self._relay_server:
            self._terminate_threads()
        self._relay_server = RelayServer(("localhost", 0), RelayTCPHandler)
        self.relay_server_thread = Thread(
            target=self._relay_server.serve_forever
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
                    r"cnf\((\w+),\w+,\((.*)\),"
                    + r"inference\((.*),.*,\[(.*)\]\)\)\.",
                    raw_clause,
                )[
                    0
                ]
            except IndexError:
                label, literals = re.findall(
                    r"cnf\((\w+),\w+,\((.*)\),file\(.*\)\)\.",
                    raw_clause,
                )[0]
                inference_rule, inference_parents = "input", None
            self.state.clauses[label] = {
                "literals": literals,
                "label": label,
                "role": "lemma",
                "birth_step": dict_clause["clause_features"]["born"] - 1,
                "inference_rule": inference_rule,
                "inference_parents": tuple(inference_parents.split(","))
                if inference_parents is not None
                else (),
            }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, Any]]:
        """
        Reset the environment.

        :param seed: seed for compatibility
        :param options: options for compatibility
        :returns: observations and info
        """
        super().reset(seed=seed)
        self._restart_relay_server()
        self.iprover_process = _iprover_start(
            self.relay_server.server_address[1],
            self.get_task(),
            self.prover_binary_path,
        )
        data = self._get_json_data()
        self._parse_iprover_requests(data)
        return tuple(self.state.clauses.values()), {}

    def _get_json_data(self) -> List[Dict[str, Any]]:
        json_data = [{"tag": "None"}]
        while json_data[-1]["tag"] not in {
            QUERY_END_MESSAGE,
            SESSION_END_MESSAGE,
        }:
            parsed_json = self.relay_server.input_queue.get()
            self.relay_server.input_queue.task_done()
            json_data.append(parsed_json)
        return json_data[1:]

    def _parse_iprover_requests(
        self, iprover_requests: List[Dict[str, Any]]
    ) -> None:
        for iprover_request in iprover_requests:
            if "clauses" in iprover_request:
                self._parse_batch_clauses(iprover_request["clauses"])

    def _do_deductions(self, action: np.int64) -> None:
        if action < len(self.state.clauses):
            given_clause_label = list(self.state.clauses.keys())[action]
            iprover_scores_message = {
                "tag": "given_clause_res",
                "passive_is_empty": False,
                "given_clause": int(given_clause_label[2:]),
            }
            relayed_scores_message = bytes(
                json.dumps({"tag": "server_queries_end"})
                + "\n\x00\n"
                + json.dumps(iprover_scores_message)
                + "\n\x00\n",
                "utf8",
            )
            self.relay_server.output_queue.put(relayed_scores_message)
            data = self._get_json_data()
            self._parse_iprover_requests(data)

    def close(self) -> None:
        """Stop relay server."""
        self._terminate_threads()

    @property
    def relay_server(self) -> RelayServer:
        """
        Return the relay server object.

        :raises ValueError: if called before ``reset``
        """
        if self._relay_server:
            return self._relay_server
        raise ValueError("run ``reset`` first!")

    def on_truncated(self) -> None:
        """Terminate threads."""
        self._terminate_threads()

    def _terminate_threads(self) -> None:
        if self._relay_server:
            self._relay_server.shutdown()
        if self.relay_server_thread:
            self.relay_server_thread.join()
        if self.iprover_process:
            try:
                self.iprover_process.terminate()
            except ProcessLookupError:  # pragma: no cover
                pass
