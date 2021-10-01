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

==========================================
 Welcome to Saturation Gym documentation!
==========================================

``gym-saturation`` is an `OpenAI Gym`_ environment for reinforcement learning (RL) agents capable of proving theorems. Currently, only theorems in `CNF`_ sublanguage of `TPTP`_ are supported. ``gym-saturation`` implements the 'given clause' algorithm (similar to one used in `Vampire`_ and `E Prover`_). Although, being written in Python, ``gym-saturation`` is closer to `PyRes`_. In contrast to monolithic architecture of a typical ATP, ``gym-saturation`` gives different agents opportunities to select clauses themselves and train from their experience. Combined with a particular agent, ``gym-saturation`` can work as an Automated Theorem Prover (ATP).

``gym-saturation`` can be interesting for RL practicioners willing to apply their experience to theorem proving without coding all the logic-related stuff themselves. It also can be useful for automated deduction researchers who want to create an RL-empowered ATP.

How to install
==============

The best way to install this package is to use ``pip``::

    pip install gym-saturation
    
How to use
==========

See `the notebook`_ for more information.

How to contribute
=================

`Pull requests`_ are welcome. To start::

    git clone https://github.com/inpefess/gym-saturation
    cd gym-saturation
    # activate python virtual environment with Python 3.6+
    pip install -U pip
    pip install -U setuptools wheel poetry
    poetry install
    # recommended but not necessary
    pre-commit install

To check the code quality before creating a pull request, one might run the script ``show_report.sh``. It locally does nearly the same as the CI pipeline after the PR is created.

Reporting issues or problems with the software
==============================================

Questions and bug reports are welcome on `the tracker`_. 

    
.. toctree::
   :maxdepth: 2
   :caption: Contents:
	     
   what-is-going-on
   testing-a-policy
   package-documentation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _OpenAI Gym: https://gym.openai.com/
.. _PyRes: https://github.com/eprover/PyResthe
.. _the notebook: https://github.com/inpefess/gym-saturation/blob/master/examples/example.ipynb
.. _the tracker: https://github.com/inpefess/gym-saturation/issues
.. _CNF: https://en.wikipedia.org/wiki/Clausal_normal_form
.. _TPTP: http://www.tptp.org/
.. _Vampire: https://github.com/vprover/vampire
.. _E Prover: https://github.com/eprover/eprover
.. _Pull requests: https://github.com/inpefess/gym-saturation/pulls
