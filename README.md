[![PyPI version](https://badge.fury.io/py/gym-saturation.svg)](https://badge.fury.io/py/gym-saturation) [![CircleCI](https://circleci.com/gh/inpefess/gym-saturation.svg?style=svg)](https://circleci.com/gh/inpefess/gym-saturation) [![Documentation Status](https://readthedocs.org/projects/gym-saturation/badge/?version=latest)](https://gym-saturation.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/inpefess/gym-saturation/branch/master/graph/badge.svg)](https://codecov.io/gh/inpefess/gym-saturation)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/inpefess/gym-saturation/HEAD?labpath=example.ipynb)

# gym-saturation

`gym-saturation` is an [OpenAI Gym](https://gym.openai.com/) environment for reinforcement learning (RL) agents capable of proving theorems. Currently, only theorems written in [TPTP library](http://tptp.org) formal language in clausal normal form (CNF) are supported. `gym-saturation` implements the 'given clause' algorithm (similar to one used in [Vampire](https://github.com/vprover/vampire) and [E Prover](https://github.com/eprover/eprover)). Being written in Python, `gym-saturation` was inspired by [PyRes](https://github.com/eprover/PyRes). In contrast to monolithic architecture of a typical Automated Theorem Prover (ATP), `gym-saturation` gives different agents opportunities to select clauses themselves and train from their experience. Combined with a particular agent, `gym-saturation` can work as an ATP.

`gym-saturation` can be interesting for RL practitioners willing to apply their experience to theorem proving without coding all the logic-related stuff themselves. It also can be useful for automated deduction researchers who want to create an RL-empowered ATP.

# How to Install

The best way to install this package is to use `pip`:

```sh
pip install gym-saturation
```

One can also run it in a Docker container:

```sh
docker build -t gym-saturation https://github.com/inpefess/gym-saturation.git
docker run -it --rm -p 8888:8888 gym-saturation jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser
```

# How to use

See [the notebook](https://github.com/inpefess/gym-saturation/blob/master/examples/example.ipynb) or run it in [Binder](https://mybinder.org/v2/gh/inpefess/gym-saturation/HEAD?labpath=example.ipynb) for more information.

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

All the tests in this package are [doctests](https://docs.python.org/3/library/doctest.html). One can run them with the following command:

```sh
pytest --doctest-modules gym-saturation
```

To check the code quality before creating a pull request, one might run the script `show_report.sh`. It locally does nearly the same as the CI pipeline after the PR is created.

# Reporting issues or problems with the software

Questions and bug reports are welcome on [the tracker](https://github.com/inpefess/gym-saturation/issues). 

# More documentation

More documentation can be found [here](https://gym-saturation.readthedocs.io/en/latest).
