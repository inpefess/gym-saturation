# Copyright 2023 Boris Shminke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# noqa: D205, D400
"""
Profile the Environment using a simple agent
============================================
"""
import cProfile
import io
import os
import pstats
from pstats import SortKey

import gymnasium as gym

from gym_saturation.agent_testing import RandomAgent, agent_testing_report
from gym_saturation.wrappers.ast2vec_wrapper import AST2VecWrapper


def _print_report(profiler: cProfile.Profile) -> None:
    profiler_report = io.StringIO()
    profiler_statistics = pstats.Stats(
        profiler, stream=profiler_report
    ).sort_stats(SortKey.TIME)
    profiler_statistics.print_stats(30)
    print(profiler_report.getvalue())


def run_profiler(use_ast2vec: bool = False) -> None:
    """
    Run profiler.

    :param use_ast2vec: whether to embed clauses
    """
    profiler = cProfile.Profile()
    profiler.enable()
    env = gym.make(
        "Vampire-v0",
        max_clauses=1000,
        problem_list=[
            os.path.join(
                os.environ["HOME"],
                "data",
                "TPTP-v8.1.2",
                "Problems",
                "GRP",
                "GRP001-1.p",
            )
        ],
    )
    if use_ast2vec:
        env = AST2VecWrapper(
            env,
            features_num=256,
        )
    iters = 1 if use_ast2vec else 10
    for _ in range(iters):
        agent_testing_report(
            env=env,  # type: ignore
            agent=RandomAgent(),
        )
    profiler.disable()
    _print_report(profiler)


if __name__ == "__main__":
    run_profiler(use_ast2vec=False)
