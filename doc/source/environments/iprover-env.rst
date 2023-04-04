..
  Copyright 2023 Boris Shminke

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

.. _iprover_env:

###########
IProverEnv
###########

``IProverEnv`` is an environment for guiding the choice of a given clause in the saturation loop of the `iProver <https://gitlab.com/korovin/iprover>`__.

.. csv-table::
   
   Action Space, ``Dicscrete(n)``
   Observation Space, "``Dict('action_mask': Box(0, 1, (n,), int8), 'real_obs': Sequence(Clause(n), stack=False))``"
   import, ``import gym_saturation; gymnasium.make("iProver-v0")``

See :ref:`saturation_env` for details on the observation and action spaces.

Description
************

iProver [1]_ is an award-winning automated theorem prover implemented in OCaml.

	
For Action Space, Observation Space, Starting State, Rewards, Episode End, and Information
*******************************************************************************************

See :ref:`saturation_env`

Arguments
**********

.. code-block:: python

   import gymnasium
    
   gymnasium.make(
       "iProver-v0",
       max_clauses=1000,
       render_mode="human",
       port_pair=None,
       iprover_binary_path="iproveropt"
   )

``max_clauses=1000``: the size ``n`` of the action space.

``render_mode="human"``: either ``ansi`` (return the clauses from the current proof state in the TPTP format) or ``human`` (print the ``ansi`` rendering to the standard output)

``port_pair=None``: a pair of ephemeral ports for the relay server. iProver will connect to the first port, a port to listen for agent's connection is the second one. If not set, port numbers are assigned at random

``iprover_binary_path="iproveropt"``: the path to iProver binary (supposed to be on the ``$PATH`` by default)

References
***********

.. [1] Duarte, A., Korovin, K. (2020). Implementing Superposition in iProver (System Description). In: Peltier, N., Sofronie-Stokkermans, V. (eds) Automated Reasoning. IJCAR 2020. Lecture Notes in Computer Science(), vol 12167. Springer, Cham. `<https://doi.org/10.1007/978-3-030-51054-1_24>`__

Version History
****************

* v0: Initial version release
