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

Variable
=========
.. autoclass:: gym_saturation.grammar.Variable
   :members:

Function
=========
.. autoclass:: gym_saturation.grammar.Function
   :members:		   

Predicate
==========
.. autoclass:: gym_saturation.grammar.Predicate
   :members:

Term
=====
.. autoclass:: gym_saturation.grammar.Term
   :members:		   

Literal
========
.. autoclass:: gym_saturation.grammar.Literal
   :members:		   

Clause
=======
.. autoclass:: gym_saturation.grammar.Clause
   :members:

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
.. autofunction:: gym_saturation.logic_ops.resolution.resolution
.. autofunction:: gym_saturation.logic_ops.resolution.all_possible_resolutions

Paramodulation
===============
.. autofunction:: gym_saturation.logic_ops.paramodulation.paramodulation
.. autofunction:: gym_saturation.logic_ops.paramodulation.all_possible_paramodulations

utils
======
.. autofunction:: gym_saturation.logic_ops.utils.deduplicate
.. autofunction:: gym_saturation.logic_ops.utils.is_subterm
.. autofunction:: gym_saturation.logic_ops.utils.get_variable_list
.. autofunction:: gym_saturation.logic_ops.utils.reindex_variables
.. autofunction:: gym_saturation.logic_ops.utils.is_tautology
.. autofunction:: gym_saturation.logic_ops.utils.clause_length
.. autofunction:: gym_saturation.logic_ops.utils.term_length
.. autofunction:: gym_saturation.logic_ops.utils.clause_in_a_list

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

.. autofunction:: gym_saturation.parsing.json_grammar.dict_to_clause

TPTP Parser
============
.. autoclass:: gym_saturation.parsing.tptp_parser.TPTPParser
   :members:

Environments
*************

Saturation Environment
=======================
.. autoclass:: gym_saturation.envs.saturation_env.SaturationEnv
   :special-members: __init__
   :members:
