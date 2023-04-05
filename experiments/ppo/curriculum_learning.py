#   Copyright 2017-2023 The Ray Authors
#   Copyright 2023 Boris Shminke
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
This file is an edited copy of an example of parametric actions usage in the Ray RLlib: https://github.com/ray-project/ray/blob/master/rllib/examples/parametric_actions_cartpole.py

Example of handling variable length and/or parametric action spaces.

This is a toy example of the action-embedding based approach for handling large
discrete action spaces (potentially infinite in size), similar to this:

    https://neuro.cs.ut.ee/the-use-of-embeddings-in-openai-five/

This currently works with RLlib's policy gradient style algorithms
(e.g., PG, PPO, IMPALA, A2C) and also DQN.

Note that since the model outputs now include "-inf" tf.float32.min
values, not all algorithm options are supported at the moment. For example,
algorithms might crash if they don't properly ignore the -inf action scores.
Working configurations are given below.
"""
import os
from typing import Any, Dict, List, Optional

import gymnasium as gym
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.examples.models.parametric_actions_model import (
    ParametricActionsModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

from gym_saturation.envs.saturation_env import PROBLEM_FILENAME, SaturationEnv
from gym_saturation.wrappers.ast2vec_wrapper import AST2VecWrapper
from gym_saturation.wrappers.duplicate_key_obs import DuplicateKeyObsWrapper
from gym_saturation.wrappers.parametric_actions_wrapper import (
    PARAMETRIC_ACTIONS,
)


def curriculum_fn(
    train_results: Dict[str, Any],
    task_settable_env: SaturationEnv,
    env_ctx: EnvContext,
) -> List[str]:
    current_task = task_settable_env.get_task()
    problem_filename = current_task[0]
    if train_results["episode_reward_mean"] > 0.95:
        new_task_index = (
            task_settable_env.problem_list.index(problem_filename) + 1
        )
        if new_task_index < len(task_settable_env.problem_list):
            new_task = [task_settable_env.problem_list[new_task_index]]
        else:
            new_task = []
    else:
        new_task = current_task
    return new_task


class TerminatedPerFile(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: Optional[int],
        **kwargs,
    ) -> None:
        agent_id = episode.get_agents()[0]
        problem_filename = os.path.splitext(
            os.path.basename(episode._last_infos[agent_id][PROBLEM_FILENAME])
        )[0]
        episode.custom_metrics[f"terminated/{problem_filename}"] = (
            1.0 if episode.is_terminated(agent_id) else 0.0
        )


if __name__ == "__main__":
    ray.init()
    problem_list = [
        os.path.join(
            os.environ["WORK"],
            "data",
            "TPTP-v8.1.2",
            "Problems",
            "SET",
            f"SET0{s}.p",
        )
        for s in [
            "01-1",  # 19
            "03-1",  # 26
            "04-1",  # 26
            "06-1",  # 28
            "02-1",  # 40
            "08-1",  # 173
            "09-1",  # 328
            "05-1",  # 1760
            "11-1",  # 2497
            "10-1",  # 8083
            "07-1",  # 14405
        ]
    ]
    # it's a standard dimension returned by ast2vec
    EMBEDDING_DIM = 256
    # ``ParametricActionsModel`` uses a fully connected network on flattened
    # observation. Since in our case observations are clauses embeddings, the
    # input layer size will be ``EMBEDDING_DIM * MAX_CLAUSES``. We can't easily
    # decrease ``EMBEDDING_DIM``, so we keep ``MAX_CLAUSES`` small. Of course,
    # one can use a different model instead of the default one (
    # ``ParametricActionsModel``).
    MAX_CLAUSES = 200
    register_env(
        "pa_cartpole",
        # and now we register our environment instead of CartPole
        lambda _: DuplicateKeyObsWrapper(
            AST2VecWrapper(
                gym.make(
                    "Vampire-v0",
                    max_clauses=MAX_CLAUSES,
                    problem_list=problem_list,
                ),
                features_num=EMBEDDING_DIM,
            ),
            key_to_duplicate=PARAMETRIC_ACTIONS,
            new_key="cart",
        ),
    )
    config = dict(
        {
            "env": "pa_cartpole",
            "env_task_fn": curriculum_fn,
            "model": {
                "custom_model": ParametricActionsModel,
                # we pass relevant parameters to ``ParametricActionsModel``
                "custom_model_config": {
                    "true_obs_shape": (EMBEDDING_DIM * MAX_CLAUSES,),
                    "action_embed_size": EMBEDDING_DIM,
                },
            },
            "num_gpus": 0,
            "num_workers": 0,
            # standard Gymnasium env_check doesn't use action masks
            # (even for standard spaces)
            "disable_env_checking": True,
            "callbacks": TerminatedPerFile,
        },
    )
    results = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(),
        param_space=config,
    ).fit()
    ray.shutdown()
