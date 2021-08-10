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

#################
Testing a Policy
#################

Suppose you already have a trained policy implemented as Python ``Callable``. Then you can use `a policy testing script`_ in parallel like that::

  find $TPTP_HOME/Problems/*/*-*.p | parallel --bar --jobs 80% --timeout 30000% python policy_testing.py --problem_file {} --output_folder TPTP_CNF --step_limit 20

You can write your own policy testing script based on ``policy_testing.py`` by calling ``episode`` function with your policy.

.. _a policy testing script: https://github.com/inpefess/gym-saturation/blob/master/gym_saturation/policy_testing.py
