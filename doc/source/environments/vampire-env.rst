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

.. _vampire_env:

###########
VampireEnv
###########

``VampireEnv`` is an environment for guiding the choice of a given clause in the saturation loop of a `Vampire <https://vprover.github.io/>`__ prover.

.. csv-table::
   
   Action Space, ``Dicscrete(n)``
   Observation Space, "``Dict('action_mask': Box(0, 1, (n,), int8), 'real_obs': Sequence(Clause(n), stack=False))``"
   import, ``import gym_saturation; gymnasium.make("Vampire-v0")``

See :ref:`saturation_env` for details on the observation and action spaces.

Description
************

Vampire (written in C++) has won the `CASC <https://tptp.org/CASC/>`__ (automated theorem provers competition) for many years. Since we focus on guiding the saturation loop here, we don't use the Avatar [1]_.
	
For Action Space, Observation Space, Starting State, Rewards, Episode End, and Information
*******************************************************************************************

See :ref:`saturation_env`

Arguments
**********

.. code-block:: python

   import gymnasium
    
   gymnasium.make(
       "Vampire-v0",
       max_clauses=1000,
       render_mode="human",
       vampire_binary_path="vampire"
   )

``max_clauses=1000``: the size ``n`` of the action space.

``render_mode="human"``: either ``ansi`` (return the clauses from the current proof state in the TPTP format) or ``human`` (print the ``ansi`` rendering to the standard output)

``vampire_binary_path="vampire"``: the path to Vampire binary (supposed to be on the ``$PATH`` by default)

References
***********

.. [1] Voronkov, A. (2014). AVATAR: The Architecture for First-Order Theorem Provers. In: Biere, A., Bloem, R. (eds) Computer Aided Verification. CAV 2014. Lecture Notes in Computer Science, vol 8559. Springer, Cham. `<https://doi.org/10.1007/978-3-319-08867-9_46>`__

Version History
****************

* v0: Initial version release
