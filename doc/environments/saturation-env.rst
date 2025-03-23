..
  Copyright 2023-2025 Boris Shminke

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

.. _saturation_env:

##############
SaturationEnv
##############

``SaturationEnv`` is an abstract class for environments guiding the choice of a given clause in the saturation algorithm used to build automated theorem provers. It has two subclasses:

* ``VampireEnv`` is an environment for guiding the choice of a given clause in the saturation loop of a `Vampire <https://vprover.github.io/>`__ prover. Since we focus on guiding the saturation loop here, we don't use the Avatar [3]_
* ``IProverEnv`` is an environment for guiding the choice of a given clause in the saturation loop of the `iProver <https://gitlab.com/korovin/iprover>`__ [1]_
  
.. csv-table::
   
   Action Space, ``Text(256)``
   Observation Space, "``Sequence(Clause(n), stack=False)``"
   import, ``import gym_saturation; gymnasium.make("Vampire-v0")``
   import, ``import gym_saturation; gymnasium.make("iProver-v0")``

Here ``Clause(n)`` is an alias for

.. code:: python

  Dict(
      'inference_parents': Sequence(
          Text(1, 256, characters=ALPHANUMERIC_WITH_UNDERSCORE),
	  stack=False
      ),
      'inference_rule': Text(1, 256, characters=ALPHANUMERIC_WITH_UNDERSCORE),
      'label': Text(1, 256, characters=ALPHANUMERIC_WITH_UNDERSCORE),
      'literals': Text(1, 4000, characters=EXTENDED_ALPHANUMERIC),
      'role': Text(1, 256, characters=ALPHANUMERIC_WITH_UNDERSCORE)
  )

and ``EXTENDED_ALPHANUMERIC`` is ``ALPHANUMERIC_WITH_UNDERSCORE`` extended by nine special characters ``(), |~=!$``. Such a structure corresponds to clauses (logical statements) in the `TPTP <https://tptp.org>`__ language.

Description
************

The given clause (or saturation) algorithm is the basis of many contemporary provers. See [2]_ for an excellent introduction and Python code snippets. In short, the algorithm is the following one:

.. code-block:: python

    unprocessed_clauses: list[Clause] = get_preprocessed_theorem_statement()
    processed_clauses: list[Clause] = []
    while EMPTY_CLAUSE not in unprocessed_clauses and unprocessed_clauses:
        given_clause: Clause = select_given_clause(unprocessed_clauses)
        processed_clauses.append(given_clause)
        new_clauses: list[Clause] = apply_inference_rules(
            given_clause, processed_clauses
        )
        unprocessed_clauses.extend(new_clauses)
        unprocessed_clauses.remove(given_clause)

``get_preprocessed_theorem_statement`` corresponds to the environment reset, and typically includes parsing, `Skolemization <https://en.wikipedia.org/wiki/Skolem_normal_form>`__, transformation to `conjunctive normal form <https://en.wikipedia.org/wiki/Conjunctive_normal_form>`__ among other things.

``apply_inference_rules`` is a 'logic engine' of a prover corresponding to what happens during the environment's ``step``. It usually includes `resolution <https://en.wikipedia.org/wiki/Resolution_(logic)>`__, `superposition <https://en.wikipedia.org/wiki/Superposition_calculus>`__ and other nasty stuff. To guarantee the algorithm applicability, the inference rules system must be `refutation complete <https://en.wikipedia.org/wiki/Completeness_(logic)#Refutation_completeness>`__.

``select_given_clause`` is a trainable agent policy.

If the ``EMPTY_CLAUSE`` (aka falsehood or contradiction) appears among the ``unprocessed_clauses``, it means we arrived at a refutation of the original set of clauses, which gives us a proof by contradiction for our theorem. If the ``unprocessed_clauses`` becomes empty, it means we've built a counter-example for our proposition.
	
Action Space
*************

Action is a label of a given clause. Different provers use different labelling systems.
    
Observation Space
******************

An observation is a tuple of strings, containing a clause literals in the TPTP syntax, e.g. ``'mult(X, mult(Y, Z)) = mult(mult(X, Y), Z)'`` for each clause belonging to ``unprocessed_clauses`` and ``processed_clauses``

Starting State
***************

A starting state of the environment depends on a task set (a theorem to prove). If there are ``N`` unprocessed clauses in the pre-processed theorem statement, the starting state contains ``N`` strings.

By default, the task is a simple theorem from group theory:

.. include:: ../../gym_saturation/resources/TPTP-mock/Problems/TST/TST001-1.p
   :literal:

One can set another task by specifying a filename of a respective TPTP problem:

.. code:: python
	  
   env.set_task(filename)

Rewards
********

Reward is ``1.0`` after a step iff the saturation algorithm terminated at this step, and ``0.0`` otherwise.

Episode End
************

* Termination means the saturation algorithm ended with refutation
  found or satisfiability established.
* There is no default truncation condition (one can add it using
  wrappers, e.g. `TimeLimit
  <https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit>`__)

Information
************

The environment returns no additional information.

Arguments
**********

.. code-block:: python

   import gymnasium
    
   gymnasium.make(
       "Vampire-v0",  # or "iProver-v0"
       prover_binary_path="vampire",  # or "iproveropt"
   )

``prover_binary_path="vampire"`` (or ``"iproveropt"``): the path to a prover binary (supposed to be on the ``$PATH`` by default)

References
***********

.. [1] Duarte, A., Korovin, K. (2020). Implementing Superposition in iProver (System Description). In: Peltier, N., Sofronie-Stokkermans, V. (eds) Automated Reasoning. IJCAR 2020. Lecture Notes in Computer Science(), vol 12167. Springer, Cham. `<https://doi.org/10.1007/978-3-030-51054-1_24>`__

.. [2] Schulz, S., Pease, A. (2020). Teaching Automated Theorem Proving by Example: PyRes 1.2. In: Peltier, N., Sofronie-Stokkermans, V. (eds) Automated Reasoning. IJCAR 2020. Lecture Notes in Computer Science(), vol 12167. Springer, Cham. `<https://doi.org/10.1007/978-3-030-51054-1_9>`__

.. [3] Voronkov, A. (2014). AVATAR: The Architecture for First-Order Theorem Provers. In: Biere, A., Bloem, R. (eds) Computer Aided Verification. CAV 2014. Lecture Notes in Computer Science, vol 8559. Springer, Cham. `<https://doi.org/10.1007/978-3-319-08867-9_46>`__       

Version History
****************

* v0: Initial version release
