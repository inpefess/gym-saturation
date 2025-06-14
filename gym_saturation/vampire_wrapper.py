# Copyright 2021-2025 Boris Shminke
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
Vampire Wrapper
================
"""  # noqa: D205, D400

from typing import Optional

import pexpect


class VampireWrapper:
    """
    A wrapper around Vampire binary running in a manual clause selection mode.

    :param binary_path: an absolute path to Vampire binary
    :param command_line_arguments: command line arguments in one string

    .. _vampire-wrapper :

    >>> from gym_saturation.constants import (MOCK_TPTP_PROBLEM,
    ...     MOCK_TPTP_FOLDER)
    >>> vampire = VampireWrapper("vampire")
    >>> vampire.pick_a_clause("2")
    Traceback (most recent call last):
     ...
    ValueError: start solving a problem first!
    >>> result = vampire.start(MOCK_TPTP_PROBLEM, MOCK_TPTP_FOLDER)
    """

    def __init__(  # noqa: D107
        self, binary_path: str, command_line_arguments: Optional[str] = None
    ):
        self.binary_path = binary_path
        self._proc = None
        self.problem_filename: Optional[str] = None
        self.command_line_arguments = (
            " --manual_cs on --show_passive on"
            " --show_new on --time_limit 0 --avatar off "
            if command_line_arguments is None
            else command_line_arguments
        )

    def _get_stdout(self) -> tuple[tuple[str, str, str], ...]:
        result: tuple[tuple[str, str, str], ...] = ()
        self.proc.expect(
            ["Pick a clause:", "Pick a clause pair:", pexpect.EOF]
        )
        if self.proc.before is not None:
            lines = self.proc.before.decode("utf-8").split("\r\n")
            for line in lines:
                if line[:5] == "[SA] ":
                    result_type, result_body = line[5:].split(": ")
                    clause_label, clause = result_body.split(". ")
                    result += ((result_type, clause_label, clause),)
        return result

    def start(
        self, problem_filename: str, tptp_folder: str
    ) -> tuple[tuple[str, str, str], ...]:
        """
        Start Vampire in a manual mode on a given problem.

        Time limit is one day, Vampire prints everything

        :param problem_filename: full path of a TPTP problem file
        :param tptp_folder: the root folder for TPTP library
        :returns: a sequence of action type, clause number and clause
        """
        self.problem_filename = problem_filename
        if self._proc is not None:
            self._proc.close()
        self._proc = pexpect.spawn(
            f"{self.binary_path} {self.command_line_arguments} "
            f"--include {tptp_folder} {problem_filename}",
            echo=False,
        )
        # https://pexpect.readthedocs.io/en/stable/commonissues.html#timing-issue-with-send-and-sendline
        self._proc.delaybeforesend = None  # type: ignore
        return self._get_stdout()

    def pick_a_clause(
        self, clause_label: str
    ) -> tuple[tuple[str, str, str], ...]:
        """
        Select a clause and get response from Vampire.

        :param clause_label: a given clause order number
        :returns: a sequence of action type, clause number and clause
        """
        self.proc.sendline(clause_label)
        return self._get_stdout()

    @property
    def proc(self) -> pexpect.spawn:
        """
        Vampire process.

        :raises ValueError: when called before ``reset``
        """
        if self._proc is None:
            raise ValueError("start solving a problem first!")
        return self._proc

    def terminate(self) -> None:
        """Terminate Vampire process if any."""
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
