..
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

######################
Package Documentation
######################

Grammar
********

.. _Variable:

Variable
=========
.. autoclass:: gym_saturation.grammar.Variable
   :members:

.. _Function:

Function
=========
.. autoclass:: gym_saturation.grammar.Function
   :members:		   

.. _Predicate:

Predicate
==========
.. autoclass:: gym_saturation.grammar.Predicate
   :members:

.. _Term:

Term
=====
.. autoclass:: gym_saturation.grammar.Term
   :members:

.. _Proposition:

Proposition
============
.. autoclass:: gym_saturation.grammar.Proposition
   :members:

.. _Literal:

Literal
========
.. autoclass:: gym_saturation.grammar.Literal
   :members:		   

.. _Clause:

Clause
=======
.. autoclass:: gym_saturation.grammar.Clause
   :members:

Utils
******

Some general utils useful for different submodules.

.. autofunction:: gym_saturation.utils.deduplicate
.. autofunction:: gym_saturation.utils.pickle_copy

   
Logic Operations
*****************

Substitution
=============
.. autoclass:: gym_saturation.logic_ops.substitution.Substitution
   :special-members: __init__, __call__
   :members:

Unification
============
.. autofunction:: gym_saturation.logic_ops.unification.most_general_unifier

Resolution
===========
.. _resolution:
.. autofunction:: gym_saturation.logic_ops.resolution.resolution
.. autofunction:: gym_saturation.logic_ops.resolution.all_possible_resolvents

Factoring
==========
.. _factoring:

.. autofunction:: gym_saturation.logic_ops.factoring.factoring
.. autofunction:: gym_saturation.logic_ops.factoring.all_possible_factors

Paramodulation
===============
.. _paramodulation:

.. autofunction:: gym_saturation.logic_ops.paramodulation.paramodulation
.. autofunction:: gym_saturation.logic_ops.paramodulation.all_paramodulants_from_clause
.. autofunction:: gym_saturation.logic_ops.paramodulation.all_paramodulants_from_list

Reflexivity Resolution
=======================
.. _reflexivity_resolution:

.. autofunction:: gym_saturation.logic_ops.reflexivity_resolution.reflexivity_resolution
.. autofunction:: gym_saturation.logic_ops.reflexivity_resolution.all_possible_reflexivity_resolvents

utils
======
.. autofunction:: gym_saturation.logic_ops.utils.is_subproposition
.. autofunction:: gym_saturation.logic_ops.utils.get_variable_list
.. autofunction:: gym_saturation.logic_ops.utils.reindex_variables
.. autofunction:: gym_saturation.logic_ops.utils.is_tautology
.. autofunction:: gym_saturation.logic_ops.utils.clause_length
.. autofunction:: gym_saturation.logic_ops.utils.proposition_length
.. autofunction:: gym_saturation.logic_ops.utils.clause_in_a_list
.. autofunction:: gym_saturation.logic_ops.utils.subterm_by_index
.. autofunction:: gym_saturation.logic_ops.utils.replace_subterm_by_index
.. autofunction:: gym_saturation.logic_ops.utils.reduce_to_proof

Parsing
********

CNF Parser
===========
.. autoclass:: gym_saturation.parsing.cnf_parser.CNFParser
   :special-members: __default__, __default_token__
   :members:

JSON Encoder
=============
.. autoclass:: gym_saturation.parsing.json_grammar.ClauseJSONEncoder
.. _dict_to_clause:

.. autofunction:: gym_saturation.parsing.json_grammar.dict_to_clause
.. autofunction:: gym_saturation.parsing.json_grammar.clause_to_dict

TPTP Parser
============
.. autoclass:: gym_saturation.parsing.tptp_parser.TPTPParser
   :members:
.. autofunction:: gym_saturation.parsing.tptp_parser.clause_to_tptp

Environments
*************

Saturation Environment
=======================
.. autoclass:: gym_saturation.envs.saturation_env.SaturationEnv
   :special-members: __init__
   :members:

Agent Testing
***************

This module is an example of testing your own trained agent.

.. autoclass:: gym_saturation.agent_testing.Transition
   :members:
.. autofunction:: gym_saturation.agent_testing.save_final_state
.. autofunction:: gym_saturation.agent_testing.size_agent
.. autofunction:: gym_saturation.agent_testing.age_agent
.. autofunction:: gym_saturation.agent_testing.random_agent
.. autofunction:: gym_saturation.agent_testing.episode
.. autofunction:: gym_saturation.agent_testing.proof_length
.. autofunction:: gym_saturation.agent_testing.agent_testing_report
