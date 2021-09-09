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

from gym_saturation.grammar import Clause, Function, Term
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
    >>> tptp_text = (
    ...     files("gym_saturation")
    ...     .joinpath("resources/TPTP-mock/Problems/TST/TST001-1.p")
    ...     .read_text()
    ... )
    >>> tptp_parser.parse(
    ...     tptp_text,
    ...     files("gym_saturation").joinpath("resources/TPTP-mock")
    ... )
    [Clause(literals=[Literal(negated=False, atom=Predicate(name='this_is_a_test_case', arguments=[Function(name='test_constant', arguments=[])]))], label='this_is_a_test_case_1', inference_parents=['one', 'two'], inference_rule='resolution', processed=None, birth_step=None), Clause(literals=[Literal(negated=True, atom=Predicate(name='this_is_a_test_case', arguments=[Function(name='test_constant', arguments=[])]))], label='this_is_a_test_case_2', inference_parents=None, inference_rule=None, processed=None, birth_step=None), Clause(literals=[Literal(negated=False, atom=Predicate(name='=', arguments=[Function(name='test_constant', arguments=[]), Variable(name='X')]))], label='test_axiom', inference_parents=None, inference_rule=None, processed=None, birth_step=None), Clause(literals=[Literal(negated=True, atom=Predicate(name='=', arguments=[Function(name='test_constant', arguments=[]), Function(name='0', arguments=[])]))], label='test_axiom_2', inference_parents=None, inference_rule=None, processed=None, birth_step=None)]
    """

    def __init__(self):
        self.parser = Lark(
            files("gym_saturation")
            .joinpath("resources/TPTP.lark")
            .read_text(),
            start="tptp_file",
        )

    def parse(self, tptp_text: str, tptp_folder: str) -> List[Clause]:
        """
        recursively parse a string containing a TPTP problem

        :param tptp_text: a name of a problem (or axioms) file
        :param parser: a ``Lark`` parser
        :param tptp_folder: a folder containing TPTP database
        :returns: a list of clauses (including those of the axioms)
        """
        problem_tree = self.parser.parse(tptp_text)
        clauses = [
            CNFParser().transform(cnf_formula)
            for cnf_formula in problem_tree.find_data("cnf_annotated")
        ]
        for include in problem_tree.find_data("include"):
            token = include.children[0]
            if isinstance(token, Token):
                with open(
                    os.path.join(tptp_folder, token.value.replace("'", "")),
                    "r",
                    encoding="utf-8",
                ) as included_file:
                    clauses.extend(
                        self.parse(included_file.read(), tptp_folder)
                    )
        return clauses


def _term_to_tptp(term: Term) -> str:
    if isinstance(term, Function):
        arguments = [_term_to_tptp(argument) for argument in term.arguments]
        if arguments != []:
            return f"{term.name}({','.join(arguments)})"
    return term.name


def clause_to_tptp(clause: Clause) -> str:
    """
    >>> from gym_saturation.grammar import Literal, Predicate, Variable
    >>> clause = Clause([Literal(True, Predicate("this_is_a_test_case", [Function("f", [Variable("X")])]))], inference_rule="resolution", inference_parents=["one", "two"], label="clause")
    >>> TPTPParser().parse(clause_to_tptp(clause), "") == [clause]
    True

    :param clause: a logic clause object
    :returns: a TPTP representation of ``clause``
    """
    res = f"cnf({clause.label}, hypothesis, "
    for literal in clause.literals:
        res += ("~" if literal.negated else "") + (
            literal.atom.name
            + "("
            + ",".join(
                [_term_to_tptp(term) for term in literal.atom.arguments]
            )
            + ") |"
        )
    if res[-1] == "|":
        res = res[:-1]
    if clause.literals == []:
        res += "$false"
    if (
        clause.inference_parents is not None
        and clause.inference_rule is not None
    ):
        res += (
            f", inference({clause.inference_rule}, [], ["
            + ",".join(clause.inference_parents)
            + "])"
        )
    return res + ")."
