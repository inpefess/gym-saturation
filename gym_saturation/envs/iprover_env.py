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
Saturation Environment with iProver back-end
============================================
"""  # noqa: D205, D400

import asyncio
import json
import os
import re
from threading import Thread
from typing import Any

from gym_saturation.envs.saturation_env import SaturationEnv
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

    :param prover_binary_path: a path to iProver binary;
        by default, we assume it to be ``iproveropt`` and in the $PATH

    Refer to :ref:`saturation_env` for more documentation.

    >>> from gymnasium.utils.env_checker import check_env
    >>> import gymnasium as gym
    >>> env = gym.make(
    ...     "iProver-v0",
    ... ).unwrapped
    >>> env.relay_server
    Traceback (most recent call last):
     ...
    ValueError: run ``reset`` first!
    >>> check_env(env)

    episode is never truncated by default

    >>> _ = env.reset()
    >>> _, _, _, truncated, _ = env.step("c_49")
    >>> truncated
    False
    """

    def __init__(  # noqa: D107
        self,
        prover_binary_path: str = "iproveropt",
    ):
        super().__init__()
        self.prover_binary_path = prover_binary_path
        self._relay_server: RelayServer | None = None
        self.relay_server_thread: Thread | None = None
        self.iprover_process: asyncio.subprocess.Process | None = None

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
        self, batch_clauses: list[dict[str, Any]]
    ) -> tuple[tuple[str, ...], set[str]]:
        new_labels: set[str] = set()
        new_clauses: tuple[str, ...] = ()
        for dict_clause in batch_clauses:
            raw_clause = (
                dict_clause["clause"].replace("\n", "").replace(" ", "")
            )
            (label, literals, inference_record) = re.findall(
                pattern=r"cnf\((\w+),\w+,\((.*)\),(\w+\(.+\))\)\.",
                string=raw_clause,
            )[0]
            new_labels.add(label)
            if inference_record[:5] == "file(":
                new_clauses += (
                    f"cnf({label},axiom,{literals},file('input.p')).",
                )
            else:
                new_clauses += (
                    f"cnf({label},plain,{literals},"
                    f"{inference_record.replace('status(thm)', '')}).",
                )
        return new_clauses, new_labels

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
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
        new_clauses, new_labels = self._parse_iprover_requests(data)
        self._available_actions = new_labels
        return new_clauses, {}

    def _get_json_data(self) -> list[dict[str, Any]]:
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
        self, iprover_requests: list[dict[str, Any]]
    ) -> tuple[tuple[str, ...], set[str]]:
        new_labels: set[str] = set()
        new_clauses: tuple[str, ...] = ()
        for iprover_request in iprover_requests:
            if "clauses" in iprover_request:
                more_new_clauses, more_new_labels = self._parse_batch_clauses(
                    iprover_request["clauses"]
                )
                new_labels.update(more_new_labels)
                new_clauses += more_new_clauses
        return new_clauses, new_labels

    def _do_deductions(self, action: str) -> tuple[tuple[str, ...], set[str]]:
        iprover_scores_message = {
            "tag": "given_clause_res",
            "passive_is_empty": False,
            "given_clause": int(action[2:]),
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
        return self._parse_iprover_requests(data)

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
