..
  Copyright 2021-2023 Boris Shminke

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

################################
Saturation Prover as an RL task
################################

One can write theorems in a machine-readable form. This package uses the `CNF`_ sub-language of `TPTP`_. Before using the environment, you will need to download a recent TPTP archive (ca 600MB).

A statement of a theorem becomes a list of clauses. In a given clause algorithm, one divides the clauses in processed and not processed yet. Then at each step, one selects a not processed yet clause as a given clause. If it's empty (we arrived at a contradiction, i.e. found a refutation proof), the algorithm stops with success. If not, one applies all possible deduction rules to the given clause and all processed clauses. Then we add deduction results to the unprocessed set, and the given clause goes into the processed. The algorithm iterates if we didn't run out of time and unprocessed clauses.

The deduction rules applied depend on a back-end. For a deduction rules used by a Vampire Python wrapper back-end, refer to `Vampire documentation <https://github.com/vprover/vampire>`__.

In an automated theorem prover (ATP) like Vampire, for the choice of a given clause, one usually employs a clever combination of heuristics. Of course, we can imagine a learning agent in charge of choosing a given clause instead, which will help us to formulate a reinforcement learning task with the following description.

What is a State
****************

(More or less resembles `ProofState class of PyRes`_)

The environment's state is a list of logical clauses. Each clause is a list of literals and also has several `properties <https://tptp-lark-parser.readthedocs.io/en/latest/package-documentation.html#tptp_lark_parser.grammar.Clause>`__.

Literal is a predicate, negated or not. A predicate can have arguments, which can be functions or variables. Functions can have arguments, which in turn can be functions or variables.

Grammar is encoded in Python objects in a self-explanatory way (see `tptp-lark-parser <https://tptp-lark-parser.readthedocs.io>`__ for more information).

What is an Observation
***********************

An observation visible by an agent is a Python dictionary having two keys: `action_mask` and `real_obs`. Action mask is a `numpy` array of zeros and ones of some fixed length. A user can change a default value (100000) for this length by passing a `max_clauses` argument to the environment constructor. If at some step there are more than `max_clauses` clauses in the state, the environment returns ``truncated == True``. For any index in `action_mask`, if there is no clause with such an index in the state, the mask value is zero. It's also zero if the clause is marked as processed. For the indices of the clauses available to become a so-called 'given clause', the mask equals one.

`real_obs` is the state (a list of clauses). Since in Gymnasium observations have to live in some predefined space, there is a Gymnasium compatible :ref:`space class<clause_space>` for a list of clauses.

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
.. _TPTP: https://www.tptp.org/
.. _ProofState class of PyRes: https://github.com/eprover/PyRes/blob/master/saturation.py
