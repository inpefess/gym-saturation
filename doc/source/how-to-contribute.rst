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

##################
How to contribute
##################

Pull requests are welcome. To start::

    git clone https://github.com/inpefess/gym-saturation
    cd gym-saturation
    # activate python virtual environment with Python 3.6+
    pip install -U pip
    pip install -U setuptools wheel poetry
    poetry install
    # recommended but not necessary
    pre-commit install

To check the code quality before creating a pull request, one might run the script ``show_report.sh``. It locally does nearly the same as the CI pipeline after the PR is created.
