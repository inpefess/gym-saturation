..
  Copyright 2021-2022 Boris Shminke

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

Or you can use `Slurm Workload Manager <https://slurm.schedmd.com/>`__. See `an example <https://github.com/inpefess/gym-saturation/tree/master/slurm-jobs>`__ from the project's repo.
  
You can write your own agent testing script based on ``agent_testing.py`` by calling ``episode`` function with your agent as an argument.

After processing problems, you can get a report about its performance::

  import os
  from gym_saturation.agent_testing import agent_testing_report
  from glob import glob
  import sys

  sys.setrecursionlimit(10000)
  problem_list = sorted(glob(os.path.join(
      os.environ["TPTP_HOME"], "Problems", "*", "*-*.p")
  ))
  report = agent_testing_report(problem_list, "TPTP_CNF_20")

Applying different policies ``gym-saturation`` leads to the following results on all CNF problems from TPTP-v7.5.0:

.. list-table:: Numbers of problems
   :header-rows: 1

   * - 
     - size agent
     - age agent
     - size&age agent
   * - **proof found**
     - 509
     - 206
     - 688
   * - **step limit**
     - 1385
     - 35
     - 223
   * - **out of memory**
     - 148
     - 149
     - 148
   * - **5 min time out**
     - 6215
     - 7867
     - 7198
   * - **total**
     - 8257
     - 8257
     - 8257

:ref:`Size agent<size_agent>` is an agent which always selects the shortest clause.
     
:ref:`Age agent<age_agent>` is an agent which always selects the clause which arrived first to the set of unprocessed clauses ('the oldest one').
     
:ref:`Size&age agent<size_age_agent>` is an agent which selects the shortest clause five times in a row and then one time --- the oldest one.
     
'Step limit' means an agent didn't find proof after 1000 steps (the longest proof found consists of 287 steps). This can work as a 'soft timeout'.

.. _an agent testing script: https://github.com/inpefess/gym-saturation/tree/master/gym_saturation/agent_testing.py
