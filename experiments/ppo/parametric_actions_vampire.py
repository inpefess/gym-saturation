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

import argparse
import os

import gymnasium as gym
import ray
from ray import air, tune
from ray.rllib.examples.models.parametric_actions_model import (
    ParametricActionsModel,
    TorchParametricActionsModel,
)
from ray.rllib.examples.random_parametric_agent import (
    RandomParametricAlgorithm,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

from gym_saturation.wrappers.ast2vec_wrapper import AST2VecWrapper
from gym_saturation.wrappers.duplicate_key_obs import DuplicateKeyObsWrapper

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.",
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters",
    type=int,
    default=200,
    help="Number of iterations to train.",
)
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.",
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=150.0,
    help="Reward at which we stop training.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    problem_list = [
        os.path.join(
            os.environ["WORK"],
            "data",
            "TPTP-v8.1.2",
            "Problems",
            "SET",
            "SET001-1.p",
        )
    ]
    # it's a standard dimension returned by ast2vec
    EMBEDDING_DIM = 256
    # ``ParametricActionsModel`` uses a fully connected network on flattened
    # observation. Since in our case observations are clauses embeddings, the
    # input layer size will be ``EMBEDDING_DIM * MAX_CLAUSES``. We can't easily
    # decrease ``EMBEDDING_DIM``, so we keep ``MAX_CLAUSES`` small. Of course,
    # one can use a different model instead of the default one (
    # ``ParametricActionsModel``).
    MAX_CLAUSES = 20
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
            # ``ParametricActionsModel`` expects a key 'cart' (from the
            # CartPole environment) to be present in the observation
            # dictionary. We add such a key and use 'avail_actions' as its
            # value, since in case of the given clause algorithm, the clauses
            # to choose from are both actions and observations.
            new_key="cart",
            key_to_duplicate="avail_actions",
        ),
    )
    ModelCatalog.register_custom_model(
        "pa_model",
        TorchParametricActionsModel
        if args.framework == "torch"
        else ParametricActionsModel,
    )

    if args.run == "DQN":
        cfg = {
            # TODO(ekl) we need to set these to prevent the masked values
            # from being further processed in DistributionalQModel, which
            # would mess up the masking. It is possible to support these if we
            # defined a custom DistributionalQModel that is aware of masking.
            "hiddens": [],
            "dueling": False,
        }
    else:
        cfg = {}

    config = dict(
        {
            "env": "pa_cartpole",
            "model": {
                "custom_model": "pa_model",
                # we pass relevant parameters to ``ParametricActionsModel``
                "custom_model_config": {
                    "true_obs_shape": (EMBEDDING_DIM * MAX_CLAUSES,),
                    "action_embed_size": EMBEDDING_DIM,
                },
            },
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 0,
            "framework": args.framework,
            # standard Gymnasium env_check doesn't use action masks
            # (even for standard spaces)
            "disable_env_checking": True,
        },
        **cfg
    )

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.Tuner(
        RandomParametricAlgorithm if args.run == "random" else args.run,
        run_config=air.RunConfig(stop=stop, verbose=1),
        param_space=config,
    ).fit()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
