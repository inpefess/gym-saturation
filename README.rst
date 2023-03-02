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
<https://gymnasium.farama.org/>`__ environments for reinforcement learning
(RL) agents striving to prove theorems. Currently, only theorems
written in `TPTP library <https://tptp.org>`__ formal language are
supported. ``gym-saturation`` implements the ‘given clause’ algorithm
(similar to one used in `Vampire
<https://github.com/vprover/vampire>`__ and `E Prover
<https://github.com/eprover/eprover>`__).

There are two environments in ``gym-saturation``: ``VampireEnv`` and
``IProverEnv``. ``VampireEnv`` is a wrapper around a recent
``Vampire`` prover, and ``IProverEnv`` relies on an experimental
version of `iProver <http://www.cs.man.ac.uk/~korovink/iprover/>`__.

In contrast to monolithic architecture of a typical Automated Theorem
Prover (ATP), ``gym-saturation`` gives different agents opportunities
to select clauses themselves and train from their experience.
Combined with a particular agent, ``gym-saturation`` can work as an
ATP.

``gym-saturation`` can be interesting for RL practitioners willing to
apply their experience to theorem proving without coding all the
logic-related stuff themselves. It also can be useful for automated
deduction researchers who want to create an RL-empowered ATP.

How to Install
==============

.. attention:: If you want to use ``VampireEnv`` you should have a
   Vampire binary on your machine. For example, download the
   latest `release
   <https://github.com/vprover/vampire/releases/tag/v4.7>`__.

   To use ``IProverEnv``, please build iProver from source using
   `this branch
   <https://gitlab.com/korovin/iprover/-/tree/2022_sockets>`__.

The best way to install this package is to use ``pip``:

.. code:: sh

   pip install gym-saturation

Another option is to use ``conda``:

.. code:: sh

   conda install -c conda-forge gym-saturation
   
One can also run it in a Docker container (with a pre-packed
``vampire`` binary):

.. code:: sh

   docker build -t gym-saturation https://github.com/inpefess/gym-saturation.git
   docker run -it --rm -p 8888:8888 gym-saturation jupyter-lab --ip=0.0.0.0 --port=8888

How to use
==========

.. code:: python

   import gym_saturation
   import gymnasium as gym
   import os

   # get a TPTP problem file or create one yourself
   env = gym.make("Vampire-v0", problem_list=["..."])
   observation, info = env.reset()
   # an order number of a 'given clause'
   action = ...
   observation, reward, terminated, truncated, info = env.step(action)

See `the
notebook <https://github.com/inpefess/gym-saturation/blob/master/examples/example.ipynb>`__ for more information.

For more comprehensive experiments, please see the documentation `page <https://gym-saturation.readthedocs.io/en/latest/experiments.html>`__.

How to Contribute
=================

`Pull requests <https://github.com/inpefess/gym-saturation/pulls>`__ are
welcome. To start:

.. code:: sh

   git clone https://github.com/inpefess/gym-saturation
   cd gym-saturation
   # activate python virtual environment with Python 3.8+
   pip install -U pip
   pip install -U setuptools wheel poetry
   poetry install
   # recommended but not necessary
   pre-commit install
   # install vampire binary
   wget https://github.com/vprover/vampire/releases/download/v4.7/vampire4.7.zip -O vampire.zip
   unzip vampire.zip
   # then use vampire_z3_rel_static_HEAD_6295 as an argument or add it to $PATH

All the tests in this package are
`doctests <https://docs.python.org/3/library/doctest.html>`__. One can
run them with the following command:

.. code:: sh

   pytest --doctest-modules gym-saturation

To check the code quality before creating a pull request, one might run
the script ``local-build.sh``. It locally does nearly the same as the CI
pipeline after the PR is created.

Reporting issues or problems with the software
==============================================

Questions and bug reports are welcome on `the
tracker <https://github.com/inpefess/gym-saturation/issues>`__.

More documentation
==================

More documentation can be found
`here <https://gym-saturation.readthedocs.io/en/latest>`__.

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
