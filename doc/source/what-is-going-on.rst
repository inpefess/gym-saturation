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

################################
Saturation Prover as an RL task
################################

One can write theorems in a machine-readable form. This package uses the language of `TPTP`_.

In a saturation prover a statement of a theorem is transformed to a clausal normal form (`CNF`_) and becomes a list of clauses. In a given clause algorithm, one divides the clauses in processed and not processed yet. Then at each step, one selects a not processed clause as a given clause. If it's empty (we arrived at a contradiction, i.e. found a refutation proof) or if there are no more unprocessed clauses (this means there is a counter-example to our proposition) the algorithm stops with success. If not, one applies all possible deduction rules to the given clause and all processed clauses. Then we add deduction results to the unprocessed set, and the given clause goes into the processed. The algorithm iterates if we didn't run out of time and/or memory.

The deduction rules applied depend on a back-end. For a deduction rules used by a Vampire environment, refer to `Vampire documentation <https://github.com/vprover/vampire>`__. For the iProver environment, the details are in the `iProver documentation <https://gitlab.com/korovin/iprover#references>`__.

In an automated theorem prover, for the choice of a given clause, one usually employs a clever combination of heuristics. Of course, we can imagine a learning agent in charge of choosing a given clause instead, which will help us to formulate a reinforcement learning task with the following description.

State
******

The environment's state is a list of logical clauses. Each clause is a disjunction of literals and also has several properties, e.g. `birth_step` (at which step of the given clause algorithm it was added to the state).

Literal is a predicate, negated or not. A predicate can have arguments, which can be functions or variables. Functions can have arguments, which in turn can be functions or variables.

`gym-saturation` doesn't parse clauses to such a detailed level, leaving them as strings of literals connected by disjunction.

Observation
************

An observation visible by an agent is a Python dictionary having two keys: `action_mask` and `real_obs`. Action mask is a `numpy` array of zeros and ones of some fixed length. A user can change a default value for this length by passing a `max_clauses` argument to the environment constructor. If at some step there are more than `max_clauses` clauses in the state, the environment returns `truncated == True`. For any index in `action_mask`, if there is no clause with such an index in the state, the mask value is zero. It's also zero if the clause is marked as processed. For the indices of the clauses available to become a given clause, the mask equals one.

`real_obs` is the state (a tuple of clauses). After the clause is added to the state, we don't change, although the respective entries of the `action_mask` can change.

One can always use an `ObservationWrapper <https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.ObservationWrapper>`__ to parse `real_obs` to a tensor representation. `gym-saturation` provides a couple of such wrappers: a generic :ref:`ParametricActionsWrapper<parametric_actions>` for any tensor representation, :ref:`AST2VecWrapper<ast2vec_wrapper>` which can be used to get tensor representation from more or less any TorchServe model (`ast2vec <https://gitlab.com/inpefess/ast2vec>`__ by default), and :ref:`FakeBoxWrapper<fake_box>` useful for a non contextual multi-armed bandit setting.

Action
*******

Action is an index of a clause from the state. Valid actions are only indices of not processed clauses, thus having ones in a respective position in `action_mask`.

One can always use an `ActionWrapper <https://gymnasium.farama.org/api/wrappers/action_wrappers/#gymnasium.ActionWrapper>`__ to change the action space to accommodate a particular approach to the problem. `gym-saturation` provides a :ref:`AgeWeightBandit <age_weight_bandit>` wrapper for a two-armed bandit: choose the oldest clause and choose the shortest clause. Notice that since we don't parse clauses, shortest means having the shortest TPTP string representation, not having the fewest number of logic symbols.

Reward
*******

`1.0` if the proof is found (a clause with an empty list of literals is selected as an action) or there are no more unprocessed clauses (and thus, there is a counter-example disproving the original proposition). Reward equals one if and only if an episode is terminated (`terminated == True`)

`0.0` otherwise

One can change these values by post-processing a collected trajectory after each episode.

Truncated
**********

An episode is truncated if and only if we have more than `max_clauses` in the `real_obs`. One can add a `TimeLimit Wrapper <https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit>`__ to a `gym-saturation` environment to model a time-limit. If without an explicit limit on the number of step on the episode, it won't last longer than `max_clauses`, since usually, at each step of a given clause algorithm, new unprocessed clauses arrive. Even if no new clauses arrive, the existing ones are marked as processed at least one at each step, so the set of processed clauses grows continuously, so again, the episode can't be longer than `max_clauses` steps.

Terminated
***********

Termination always means a success (proof or a counter-example) and is rewarded by `1.0`.

Info
*****

At each step, the environment returns an `info` dictionary, by default containing only one key: `problem_filename`.

Important notice
*****************

Usually, saturation provers use a timeout in seconds since they work in real-time mode. Here, we live in a discrete time, so we limit a prover by the number of saturation algorithm steps taken, not wall-clock time. The same goes for the RAM limit: we use `max_clauses` to model it instead of real bytes of RAM.

.. _CNF: https://en.wikipedia.org/wiki/Clausal_normal_form
.. _TPTP: https://www.tptp.org/
