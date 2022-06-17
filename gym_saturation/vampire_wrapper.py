# Copyright 2021-2022 Boris Shminke
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
Vampire Wrapper
================
"""
from typing import Tuple

import pexpect


class VampireWrapper:
    """
    A wrapper around Vampire binary running in a manual clause selection mode.

    .. _vampire-wrapper :

    >>> import sys
    >>> if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> import os
    >>> tptp_folder = files("gym_saturation").joinpath(
    ...     os.path.join("resources", "TPTP-mock")
    ... )
    >>> tptp_problem = os.path.join(
    ...     tptp_folder, "Problems", "TST", "TST003-1.p"
    ... )
    >>> vampire = VampireWrapper("vampire")
    >>> vampire.pick_a_clause("2")
    Traceback (most recent call last):
     ...
    ValueError: start solving a problem first!
    >>> old_result = vampire.start(tptp_problem, tptp_folder)
    >>> print(old_result)
    (('input', '1', 'animal(X0) | ~carnivore(X0) [input]'),...
    >>> new_result = vampire.start(tptp_problem, tptp_folder)
    >>> old_result == new_result
    True
    >>> vampire.pick_a_clause("wrong")
    Traceback (most recent call last):
     ...
    ValueError:
    wrong
    ...
    """

    def __init__(self, binary_path: str):
        """
        Remember the path to Vampire, don't start the process yet.

        :param binary_path: an absolute path to Vampire binary
        """
        self.binary_path = binary_path
        self._proc = None

    def _get_stdout(self) -> Tuple[Tuple[str, str, str], ...]:
        result: Tuple[Tuple[str, str, str], ...] = ()
        self.proc.expect(["Pick a clause:", pexpect.EOF])
        for line in self.proc.before.decode("utf-8").split("\r\n"):
            if (
                line[:5] == "[SA] "
                or line[:12] in {"[PP] final: ", "[PP] input: "}
                or "[PP] fn def discovered: " in line
            ):
                result_type, result_body = line[5:].split(": ")
                clause_label, clause = result_body.split(". ")
                result += ((result_type, clause_label, clause),)
        if result:
            return result
        raise ValueError(self.proc.before.decode("utf-8"))

    def start(
        self, problem_filename: str, tptp_folder: str
    ) -> Tuple[Tuple[str, str, str], ...]:
        """
        Start Vampire in a manual mode on a given problem.

        Time limit is one day, Vampire prints everything

        :param problem_filename: full path of a TPTP problem file
        :param tptp_folder: the root folder for TPTP library
        :returns: a sequence of action type, clause number and clause
        """
        if self._proc is not None:
            self._proc.close()
        self._proc = pexpect.spawn(
            f"{self.binary_path} --manual_cs on --show_everything on "
            + "--time_limit 1D --avatar off "
            + f"--include {tptp_folder} {problem_filename}"
        )
        return self._get_stdout()

    def pick_a_clause(
        self, clause_label: str
    ) -> Tuple[Tuple[str, str, str], ...]:
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
