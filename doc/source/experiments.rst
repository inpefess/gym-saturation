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

##################################################
Using gym-saturation with different RL algorithms
##################################################

The experiments demonstrating `gym-saturation` compatibility with different RL algorithms are done with `Ray RLlib <https://docs.ray.io/en/latest/rllib/index.html>`__. They work with both `torch <https://pytorch.org/>`__ and `tensorflow <https://www.tensorflow.org/>`__ DNN frameworks. The experiment results can be observed by running `tensorboard --logdir ~/ray_results`.

To run the `PPO <https://arxiv.org/abs/1707.06347>`__ and `DQN <https://arxiv.org/abs/1312.5602>`__ examples, one should run an `ast2vec <https://arxiv.org/abs/2103.11614>`__ `server <https://gitlab.com/inpefess/ast2vec>`__ first.

In the experiments, we try to solve a `trivial set theory problem <https://tptp.org/cgi-bin/SeeTPTP?Category=Problems&Domain=SET&File=SET001-1.p>`__ which is known to be provable in 9 steps. The experiments are based on RLlib examples with minimal code edits. No spectacular performance is thus expected.

Thompson Sampling Multi-armed Bandit
*************************************

#. Navigate to `examples/multi-armed-bandit`
#. `pip install -r requirements.txt`
#. `python thompson_sampling.py --random_baseline`
#. `python thompson_sampling.py`

Bandit learns to prefer choosing the shortest clause, beating the random agent substantially.

PPO
****

#. Navigate to `examples/ppo`
#. `pip install -r requirements.txt`
#. `python parametric_actions_vampire.py`

PPO steadily converges to perfectly solving the problem and even finding the 9 steps proof.
   
DQN
****

#. Navigate to `examples/ppo`
#. `pip install -r requirements.txt`
#. `python parametric_actions_vampire.py --run DQN`

Since the DQN uses a replay buffer and we have the zero reward nearly all the time, the data used for training (sampled from the replay buffer) are extremely unbalanced. This makes the DQN convergence somehow unstable, but it eventually manages to find the proof most of the time.
