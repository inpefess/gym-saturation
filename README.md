[![PyPI version](https://badge.fury.io/py/gym-saturation.svg)](https://badge.fury.io/py/gym-saturation) [![CircleCI](https://circleci.com/gh/inpefess/gym-saturation.svg?style=svg)](https://circleci.com/gh/inpefess/gym-saturation) [![Documentation Status](https://readthedocs.org/projects/gym-saturation/badge/?version=latest)](https://gym-saturation.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/inpefess/gym-saturation/branch/master/graph/badge.svg)](https://codecov.io/gh/inpefess/gym-saturation)

# gym-saturation

`gym-saturation` is an [OpenAI Gym](https://gym.openai.com/) environment for reinforcement learning (RL) agents capable of proving theorems. Currently, only theorems in CNF sublanguage of [TPTP](http://tptp.org) are supported. `gym-saturation` implements the 'given clause' algorithm (similar to one used in [Vampire](https://github.com/vprover/vampire) and [E Prover](https://github.com/eprover/eprover)). Although, being written in Python, `gym-saturation` is closer to [PyRes](https://github.com/eprover/PyRes). In contrast to monolithic architecture of a typical ATP, `gym-saturation` gives different agents opportunities to select clauses themselves and train from their experience. Combined with a particular agent, `gym-saturation` can work as an Automated Theorem Prover (ATP).

`gym-saturation` can be interesting for RL practicioners willing to apply their experience to theorem proving without coding all the logic-related stuff themselves. It also can be useful for automated deduction researchers who want to create an RL-empowered ATP.

# How to Install

The best way to install this package is to use `pip`:

```sh
pip install gym-saturation
```

# How to use

See [the notebook](https://github.com/inpefess/gym-saturation/blob/master/examples/example.ipynb) for more information.

# How to Contribute

[Pull requests](https://github.com/inpefess/gym-saturation/pulls) are welcome. To start:

```sh
git clone https://github.com/inpefess/gym-saturation
cd gym-saturation
# activate python virtual environment with Python 3.6+
pip install -U pip
pip install -U setuptools wheel poetry
poetry install
# recommended but not necessary
pre-commit install
```

To check the code quality before creating a pull request, one might run the script `show_report.sh`. It locally does nearly the same as the CI pipeline after the PR is created.


# Reporting issues or problems with the software

Questions and bug reports are welcome on [the tracker](https://github.com/inpefess/gym-saturation/issues). 

# More documentation

More documentation can be found [here](https://gym-saturation.readthedocs.io/en/latest).
