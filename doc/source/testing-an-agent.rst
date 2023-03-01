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

#################
Testing an Agent
#################

Suppose you already have a trained agent implemented. Then you can use `an agent testing script`_ in parallel like that::

  find $TPTP_HOME/Problems/*/*-*.p | parallel --bar --jobs 80% --timeout 30000% python agent_testing.py --problem_file {} --output_folder TPTP_CNF --step_limit 20

Or you can use `Slurm Workload Manager <https://slurm.schedmd.com/>`__. See `an example <https://github.com/inpefess/gym-saturation/tree/master/slurm-jobs>`__ from the project's repository.

You can write your own agent testing script based on ``agent_testing.py`` by calling ``episode`` function with your agent as an argument.

After processing problems, you can get a report about its performance::

  import os
  from gym_saturation.agent_testing import agent_testing_report
  from glob import glob
  import sys

  problem_list = sorted(glob(os.path.join(
      os.environ["TPTP_HOME"], "Problems", "*", "*-*.p")
  ))
  report = agent_testing_report(problem_list, "...")

:ref:`Weight agent<weight_agent>` is an agent which always selects the shortest clause.

:ref:`Age agent<age_agent>` is an agent which always selects the clause which arrived first to the set of unprocessed clauses ('the oldest one').

:ref:`Age&Weight agent<age_weight_agent>` is an agent which selects the oldest clause several times in a row and then several (probably a different number of) times --- the shotest one.
     
.. _an agent testing script: https://github.com/inpefess/gym-saturation/tree/master/gym_saturation/agent_testing.py
