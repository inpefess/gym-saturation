"""
Copyright 2021 Boris Shminke

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import sys
from typing import List

from lark import Lark, Token

from gym_saturation.grammar import Clause
from gym_saturation.parsing.cnf_parser import CNFParser

if sys.version_info.major == 3 and sys.version_info.minor == 9:
    # pylint: disable=no-name-in-module, import-error
    from importlib.resources import files  # type: ignore
else:
    from importlib_resources import files  # pylint: disable=import-error


# pylint: disable=too-few-public-methods
class TPTPParser:
    """
    >>> tptp_parser = TPTPParser()
    >>> tptp_parser.parse(
    ...     files("gym_saturation")
    ...     .joinpath("resources/TPTP-mock/Problems/TST/TST001-1.p"),
    ...     files("gym_saturation").joinpath("resources/TPTP-mock")
    ... )
    [Clause(literals=[Literal(negated=False, atom=Predicate(name='this_is_a_test_case', arguments=[Function(name='test_constant', arguments=[])]))], label='test_formula', inference_parents=None, processed=None, birth_step=None), Clause(literals=[Literal(negated=True, atom=Predicate(name='this_is_a_test_case', arguments=[Function(name='test_constant', arguments=[])]))], label='test_formula', inference_parents=None, processed=None, birth_step=None), Clause(literals=[Literal(negated=False, atom=Predicate(name='=', arguments=[Function(name='test_constant', arguments=[]), Variable(name='X')]))], label='test_axiom', inference_parents=None, processed=None, birth_step=None), Clause(literals=[Literal(negated=False, atom=Predicate(name='!=', arguments=[Function(name='test_constant', arguments=[]), Function(name='0', arguments=[])]))], label='test_axiom_2', inference_parents=None, processed=None, birth_step=None)]
    """

    def __init__(self):
        self.parser = Lark(
            files("gym_saturation")
            .joinpath("resources/TPTP.lark")
            .read_text(),
            start="tptp_file",
        )

    def parse(self, filename: str, tptp_folder: str) -> List[Clause]:
        """
        recursively parse a TPTP problem (or axioms) file

        :param filename: a name of a problem (or axioms) file
        :param parser: a ``Lark`` parser
        :param tptp_folder: a folder containing TPTP database
        :returns: a list of clauses (including those of the axioms)
        """
        with open(filename, "r") as problem_file:
            problem_tree = self.parser.parse(problem_file.read())
        clauses = [
            CNFParser().transform(cnf_formula)
            for cnf_formula in problem_tree.find_data("cnf_annotated")
        ]
        for include in problem_tree.find_data("include"):
            token = include.children[0]
            if isinstance(token, Token):
                clauses.extend(
                    self.parse(
                        os.path.join(
                            tptp_folder, token.value.replace("'", "")
                        ),
                        tptp_folder,
                    )
                )
        return clauses
