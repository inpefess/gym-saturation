# Copyright 2021-2022 Boris Shminke

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CNF Parser
===========
"""
import dataclasses
from operator import itemgetter

from lark import Transformer

from gym_saturation.grammar import (
    Clause,
    Function,
    Literal,
    Predicate,
    Variable,
)


class CNFParser(Transformer):
    """
    a parser for ``<cnf_formula>`` from Lark parse tree
    methods are not typed since nobody calls them directly

    >>> import sys
    >>> import os
    >>> if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    ...     from importlib.resources import files
    ... else:
    ...     from importlib_resources import files
    >>> from lark import Lark
    >>> parser = Lark(
    ...     files("gym_saturation")
    ...     .joinpath(os.path.join("resources", "TPTP.lark"))
    ...     .read_text(),
    ...     start="tptp_file"
    ... )
    >>> CNFParser().transform(parser.parse('''
    ...    cnf(test, axiom, f(X, g(Y), h(Z, c1)) = f(X, Y, c2)
    ...    | ~ better(f(X), g(Y)) | $false | this_is_a_test_case).
    ... '''))
    cnf(test, axiom, f(X,g(Y),h(Z,c1)) = f(X,Y,c2) | ~better(f(X), g(Y)) | $false() | this_is_a_test_case()).
    """

    def __default_token__(self, token):
        return token.value

    def __default__(self, data, children, meta):
        if len(children) == 1:
            return children[0]
        return children

    @staticmethod
    def _function(children):
        """
        a functional symbol with arguments
        """
        if len(children) > 1:
            return Function(children[0], tuple(children[1]))
        return Function(children[0], ())

    def fof_defined_plain_formula(self, children):
        """
        <fof_defined_plain_formula> :== <defined_proposition> | <defined_predicate>(<fof_arguments>)
        """
        return self._predicate(children)

    def fof_plain_term(self, children):
        """
        <fof_plain_term>       ::= <constant> | <functor>(<fof_arguments>)
        """
        return self._function(children)

    def fof_defined_term(self, children):
        """
        <fof_defined_term>     ::= <defined_term> | <fof_defined_atomic_term>
        """
        return self._function(children)

    @staticmethod
    def variable(children):
        """
        <variable>             ::= <upper_word>

        a variable (supposed to be universally quantified)
        """
        return Variable(children[0])

    @staticmethod
    def fof_arguments(children):
        """
        <fof_arguments>        ::= <fof_term> | <fof_term>,<fof_arguments>

        a list of arguments, organised in pairs
        """
        result = ()
        for item in children:
            if not isinstance(item, (Variable, Function)):
                result = result + tuple(item)
            else:
                result = result + (item,)
        return result

    @staticmethod
    def literal(children):
        """
        <literal>              ::= <fof_atomic_formula> | ~ <fof_atomic_formula> | <fof_infix_unary>

        a literal is a possible negated predicate
        """
        if children[0] == "~":
            return Literal(True, children[1])
        if isinstance(children[0], Predicate):
            if children[0].name == "!=":
                return Literal(
                    negated=True,
                    atom=Predicate(name="=", arguments=children[0].arguments),
                )
        return Literal(False, children[0])

    @staticmethod
    def _predicate(children):
        """predicates are atomic formulae"""
        if len(children) > 1:
            return Predicate(children[0], tuple(children[1]))
        return Predicate(children[0], ())

    def fof_plain_atomic_formula(self, children):
        """
        <fof_plain_atomic_formula> :== <proposition> | <predicate>(<fof_arguments>)
        """
        return self._predicate(children)

    @staticmethod
    def fof_defined_infix_formula(children):
        """
        <fof_defined_infix_formula> ::= <fof_term> <defined_infix_pred> <fof_term>

        some predicates are in the infix form, so we translate to them prefix
        """
        return Predicate(children[1], (children[0], children[2]))

    @staticmethod
    def fof_infix_unary(children):
        """
        <fof_infix_unary>      ::= <fof_term> <infix_inequality> <fof_term>

        some predicates are in the infix form, so we translate to them prefix
        """
        return Predicate(children[1], (children[0], children[2]))

    @staticmethod
    def disjunction(children):
        """
        <disjunction>          ::= <literal> | <disjunction> <vline> <literal>

        basic clause structure
        """
        if len(children) == 1:
            return Clause(tuple(children))
        literals = ()
        for item in (children[0], children[2]):
            if isinstance(item, Clause):
                literals = literals + item.literals
            else:
                literals = literals + (item,)
        return Clause(literals)

    @staticmethod
    def _parse_inference_parents(inference_parents):
        if isinstance(inference_parents[0], list):
            clause_inference_parents = tuple(
                map(
                    itemgetter(0),
                    inference_parents,
                )
            )
        else:
            clause_inference_parents = (inference_parents[0],)
        return clause_inference_parents

    def cnf_annotated(self, children):
        """
        <cnf_annotated>        ::= cnf(<name>,<formula_role>,<cnf_formula> <annotations>).

        annotated CNF formula (clause)
        """
        clause = children[2]
        inference_rule = None
        inference_parents = None
        if isinstance(children[3], list):
            for annotation in children[3]:
                if isinstance(annotation, dict):
                    if "inference_record" in annotation:
                        inference_rule = annotation["inference_record"][0]
                        inference_parents = self._parse_inference_parents(
                            annotation["inference_record"][1]
                        )
        return dataclasses.replace(
            clause,
            label=children[0],
            role=children[1],
            inference_rule=inference_rule,
            inference_parents=inference_parents,
        )

    @staticmethod
    def inference_record(children):
        """
        <inference_record>     :== inference(<inference_rule>,<useful_info>,
        <inference_parents>)
        """
        return {"inference_record": (children[0], children[2])}

    @staticmethod
    def annotations(children):
        """
        <annotations>          ::= ,<source><optional_info> | <null>
        """
        if isinstance(children, list):
            if len(children) == 1:
                return tuple(children[0])
        return children

    @staticmethod
    def source(children):
        """
        <source>               ::= <general_term>
        <source>               :== <dag_source> | <internal_source> |
        <external_source> | unknown | [<sources>]
        """
        if isinstance(children, list):
            if len(children) == 1:
                return children[0]
        return children

    @staticmethod
    def dag_source(children):
        """
        <dag_source>           :==  <inference_record> | <name>
        """
        if isinstance(children, list):
            if len(children) == 1:
                return children[0]
        return children
