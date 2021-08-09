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

#################  
What is going on
#################

One can write theorems in a machine-readable form. This package uses the `CNF`_ sublanguage of `TPTP`_. Before using the environment, you will need to download a recent TPTP archive (ca 600MB).

A statement of a theorem becomes a list of clauses. In a given clause algorithm, one divides the clauses in processed and not processed yet. Then at each step, one selects a not processed yet clause as a given clause. If it's empty (we arrived at a contradiction, i.e. found a refutation proof), the algorithm stops with success. If not, one applies all possible deduction rules to the given clause and all processed clauses. Then we add deduction results to the unprocessed set, and the given clause goes into the processed. The algorithm iterates if we didn't run out of time and unprocessed clauses.

The deduction rules are the following (this deductive system is known to be refutation complete):

* `resolution`_
* `factoring`_
* `paramodulation`_
* reflexivity resolution (in fact, a paramodulation variant)

For the choice of a given clause, one usually employs a clever combination of heuristics. Of course, we can reformulate the same process as a reinforcement learning task.

What is a State
****************

(More or less resembles `ProofState class of PyRes`_)

The environment's state is a list of logical clauses. Each clause is a list of literals and also has several properties:

* ``label`` --- comes from the problem file or starts with ``inferred_`` if inferred during the episode
* ``processed`` --- boolean value splitting clauses into unprocessed and processed ones; in the beginning, everything is not processed
* ``birth_step`` --- a number of the step when the clause appeared in the unprocessed set; clauses from the problem have ``birth_step`` zero
* ``inference_parents`` --- a list of labels from which the clause was inferred. For clauses from the problem statement, this list is empty.

Literal is a predicate, negated or not. A predicate can have arguments, which can be functions or variables. Functions can have arguments, which in turn can be functions or variables.

Grammar is encoded in Python objects in a self-explanatory way. Each grammar object is a dictionary with an obligatory key ``class`` (``Clause``, ``Literal``, ``Predicate``, ``Function``, ``Variable``), and other keys representing this object's properties (such as being negated or having a list of arguments). To parse these JSON representation into package's inner representation, use ``gym_saturation.parsing.json_grammar.dict_to_clause``.

What is an Action
******************

Action is an index of a clause from the state. Valid actions are only indices of not processed clauses.

What is a Reward
*****************

``1.0`` if the proof is found (a clause with an empty list of literals is selected as an action).

``0.0`` otherwise

Important notice
*****************

Usually, saturation provers use a timeout in seconds since they work in real-time mode. Here, we live in a discrete time, so we limit a prover by the number of saturation algorithm steps taken, not wall-clock time.

.. _CNF: https://en.wikipedia.org/wiki/Clausal_normal_form
.. _TPTP: http://www.tptp.org/
.. _ProofState class of PyRes: https://github.com/eprover/PyRes/blob/master/saturation.py
.. _resolution: https://en.wikipedia.org/wiki/Resolution_(logic)#Resolution_in_first_order_logic
.. _factoring: https://en.wikipedia.org/wiki/Resolution_(logic)#Factoring
.. _paramodulation: https://en.wikipedia.org/wiki/Resolution_(logic)#Paramodulation
