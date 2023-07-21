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

|PyPI version|\ |Anaconda|\ |CircleCI|\ |Documentation Status|\ |codecov|\ |JOSS|

gym-saturation
==============

``gym-saturation`` is a collection of `Gymnasium
<https://gymnasium.farama.org/>`__ environments for reinforcement
learning (RL) agents guiding saturation-style automated theorem
provers (ATPs) based on the `given clause algorithm
<https://royalsocietypublishing.org/doi/10.1098/rsta.2018.0034#d3e468>`__.

There are two environments in ``gym-saturation`` following the same
API: `SaturationEnv
<https://gym-saturation.readthedocs.io/en/latest/environments/saturation-env.html>`__:
``VampireEnv`` --- for `Vampire
<https://github.com/vprover/vampire>`__ prover, and ``IProverEnv``
--- for `iProver <https://gitlab.com/korovin/iprover/>`__.

``gym-saturation`` can be interesting for RL practitioners willing to
apply their experience to theorem proving without coding all the
logic-related stuff themselves.

In particular, ATPs serving as ``gym-saturation`` backends
incapsulate parsing the input formal language (usually, one of the
`TPTP <https://tptp.org/>`__ (Thousands of Problems for Theorem
Provers) library), transforming the input formulae to the `clausal
normal form
<https://en.wikipedia.org/wiki/Conjunctive_normal_form>`__, and logic
inference using rules such as `resolution
<https://en.wikipedia.org/wiki/Resolution_(logic)>`__ and
`superposition
<https://en.wikipedia.org/wiki/Superposition_calculus>`__.

How to Install
==============

.. attention:: If you want to use ``VampireEnv`` you should have a
   Vampire binary on your machine. For example, download the
   latest `release
   <https://github.com/vprover/vampire/releases/tag/v4.8casc2023>`__.

   To use ``IProverEnv``, please download a stable iProver 
   `release
   <https://gitlab.com/inpefess/iprover/-/releases/2023.04.10>`__ or build it from `this commit <https://gitlab.com/korovin/iprover/-/commit/9504e78f60262c394b564678201cb7d2f8a4bbea>`__.

The best way to install this package is to use ``pip``:

.. code:: sh

   pip install gym-saturation

Another option is to use ``conda``:

.. code:: sh

   conda install -c conda-forge gym-saturation
   
One can also run it in a Docker container (pre-packed with
``vampire`` and ``iproveropt`` binaries):

.. code:: sh

   docker build -t gym-saturation https://github.com/inpefess/gym-saturation.git
   docker run -it --rm -p 8888:8888 gym-saturation jupyter-lab --ip=0.0.0.0 --port=8888

How to use
==========

One can use ``gym-saturation`` environments as any other Gymnasium environment:

.. code:: python

  import gym_saturation
  import gymnasium

  env = gymnasium.make("Vampire-v0")  # or "iProver-v0"
  # skip this line to use the default problem
  env.set_task("a-TPTP-problem-filename")
  observation, info = env.reset()
  terminated, truncated = False, False
  while not (terminated or truncated):
      # apply policy (a valid random action here)
      action = env.action_space.sample(mask=observation["action_mask"])
      observation, reward, terminated, truncated, info = env.step(action)
  env.close()

Or have a look at the basic `tutorial <https://gym-saturation.readthedocs.io/en/latest/auto_examples/plot_age_agent.html>`__.
  
For a bit more comprehensive experiments, please navigate the documentation `page <https://gym-saturation.readthedocs.io/en/latest/tutorials/experiments.html>`__.

More Documentation
==================

More documentation can be found
`here <https://gym-saturation.readthedocs.io/en/latest>`__.

Related Projects
=================

``gym-saturation`` is compatible with RL-frameworks such as `Ray
RLlib <https://docs.ray.io/en/latest/rllib/package_ref/index.html>`__
and can leverage code embeddings such as `CodeBERT
<https://github.com/microsoft/CodeBERT>`__.

Other projects using RL-guidance for ATPs include:

* `TRAIL <https://github.com/IBM/TRAIL>`__
* `FLoP <https://github.com/atpcurr/atpcurr>`__ (see `the paper <https://doi.org/10.1007/978-3-030-86059-2_10>`__ for more details)
* `lazyCoP <https://github.com/MichaelRawson/lazycop>`__ (see `the paper <https://doi.org/10.1007/978-3-030-86059-2_11>`__ for more details)

Other projects not using RL per se, but iterating a supervised
learning procedure instead:

* ENIGMA (several repos, e.g. `this one
  <https://gitlab.ciirc.cvut.cz/chvalkar/iprover-gnn-server>`__ for
  iProver; see `the paper <https://doi.org/10.29007/tp23>`__ for
  others)
* `Deepire <https://github.com/quickbeam123/deepire-paper-supplementary-materials>`__

How to Contribute
=================

Please follow `the contribution guide <https://gym-saturation.readthedocs.io/en/latest/contributing.html>`__ while adhering to `the code of conduct <https://gym-saturation.readthedocs.io/en/latest/code-of-conduct.html>`__.

How to Cite
============

If you are writing a research paper and want to cite ``gym-saturation``, please use the following `DOI <https://doi.org/10.21105/joss.03849>`__.

.. |PyPI version| image:: https://badge.fury.io/py/gym-saturation.svg
   :target: https://badge.fury.io/py/gym-saturation
.. |CircleCI| image:: https://circleci.com/gh/inpefess/gym-saturation.svg?style=svg
   :target: https://circleci.com/gh/inpefess/gym-saturation
.. |Documentation Status| image:: https://readthedocs.org/projects/gym-saturation/badge/?version=latest
   :target: https://gym-saturation.readthedocs.io/en/latest/?badge=latest
.. |codecov| image:: https://codecov.io/gh/inpefess/gym-saturation/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/inpefess/gym-saturation
.. |JOSS| image:: https://joss.theoj.org/papers/c4f36ec7331a0dde54d8c3808fbff9c3/status.svg
   :target: https://joss.theoj.org/papers/c4f36ec7331a0dde54d8c3808fbff9c3
.. |Anaconda| image:: https://anaconda.org/conda-forge/gym-saturation/badges/version.svg
   :target: https://anaconda.org/conda-forge/gym-saturation
