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

############
Basic Usage
############

Initializing Environments
**************************

Using environments from ``gym-saturation`` is very similar to using them in `Gymnasium <https://gymnasium.farama.org/>`__. You initialize an environment via:

.. code:: python

   import gym_saturation
   import gymnasium as gym

   env = gym.make("Vampire-v0")

Additional Environment API
***************************

There are two additional methods to each ``gym-saturation`` environment:

* ``set_task`` --- to specify a filename of a `TPTP <https://tptp.org/>`__ problem to solve (like in `MetaWorld <https://github.com/Farama-Foundation/Metaworld>`__ multi-task environments)
* ``get_task`` --- to look up a filename of a TPTP problem being solved (like in ``TaskSettableEnv`` in `Ray RLlib <https://docs.ray.io/en/latest/rllib/rllib-advanced-api.html#curriculum-learning>`__)
