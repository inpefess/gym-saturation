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
import json
from typing import Dict, List, Union

from gym_saturation import grammar


class ClauseJSONEncoder(json.JSONEncoder):
    """
    by default ``Clause`` is not serializable type. This class is a standard
    way to make it such. It's coupled with a function ``dict_to_clause`` to
    restore serialised objects back. Potentially that can be done in any
    language, not only Python.

    >>> this_is_a_test_case = grammar.Clause([grammar.Literal(False, grammar.Predicate("=", [grammar.Function("f", [grammar.Variable("X"), grammar.Function("g", [grammar.Variable("Y")]), grammar.Function("h", [grammar.Variable("Z"), grammar.Function("c1", [])])]), grammar.Function("f", [grammar.Variable("X"), grammar.Variable("Y"), grammar.Function("c2", [])])])), grammar.Literal(True, grammar.Predicate("better", [grammar.Variable("X"), grammar.Function("g", [grammar.Variable("Y")])]))], label="some", birth_step=0, inference_rule="resolution", inference_parents=["one", "two"], processed=False)
    >>> # this is a standard way to serialise a ``Clause``
    >>> serialised_clause = clause_to_dict(this_is_a_test_case)
    >>> # and this is how to deserialise
    >>> deserialised_clause = dict_to_clause(serialised_clause)
    >>> # sanity check
    >>> print(this_is_a_test_case == deserialised_clause)
    True
    >>> dict_to_clause({"class": "not a Clause!"})
    Traceback (most recent call last):
     ...
    ValueError: json is not a Clause: {'class': 'not a Clause!'}
    """

    @staticmethod
    def _variable_to_dict(variable: grammar.Variable) -> Dict[str, str]:
        return {"class": "Variable", "name": variable.name}

    def _function_to_dict(
        self, function: grammar.Function
    ) -> Dict[str, Union[str, list]]:
        arguments: List[Union[str, dict]] = []
        for argument in function.arguments:
            if isinstance(argument, grammar.Variable):
                arguments.append(self._variable_to_dict(argument))
            else:
                arguments.append(self._function_to_dict(argument))
        return {
            "class": "Function",
            "name": function.name,
            "arguments": arguments,
        }

    def default(self, o):
        literals = []
        for literal in o.literals:
            arguments = []
            for argument in literal.atom.arguments:
                if isinstance(argument, grammar.Variable):
                    arguments.append(self._variable_to_dict(argument))
                else:
                    arguments.append(self._function_to_dict(argument))
            literals.append(
                {
                    "class": "Literal",
                    "negated": literal.negated,
                    "atom": {
                        "class": "Predicate",
                        "name": literal.atom.name,
                        "arguments": arguments,
                    },
                }
            )
        return {
            "class": "Clause",
            "literals": literals,
            "label": o.label,
            "birth_step": o.birth_step,
            "processed": o.processed,
            "inference_parents": o.inference_parents,
            "inference_rule": o.inference_rule,
        }


def dict_to_clause(json_dict):
    """get ``Clause`` from a Python dictionary"""
    if json_dict["class"] == "Clause":
        return grammar.Clause(
            [dict_to_clause(literal) for literal in json_dict["literals"]],
            label=json_dict["label"],
            birth_step=json_dict["birth_step"],
            processed=json_dict["processed"],
            inference_parents=json_dict["inference_parents"],
            inference_rule=json_dict["inference_rule"],
        )
    if json_dict["class"] == "Literal":
        return grammar.Literal(
            json_dict["negated"], dict_to_clause(json_dict["atom"])
        )
    if json_dict["class"] == "Predicate":
        return grammar.Predicate(
            json_dict["name"],
            [dict_to_clause(argument) for argument in json_dict["arguments"]],
        )
    if json_dict["class"] == "Function":
        return grammar.Function(
            json_dict["name"],
            [dict_to_clause(argument) for argument in json_dict["arguments"]],
        )
    if json_dict["class"] == "Variable":
        return grammar.Variable(json_dict["name"])
    raise ValueError(f"json is not a Clause: {json_dict}")


def clause_to_dict(clause):
    """convert clause to JSON-like dictionary"""
    return json.loads(json.dumps(clause, cls=ClauseJSONEncoder))
