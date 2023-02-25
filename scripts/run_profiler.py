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

from gym_saturation.agent_testing import test_agent


def run_profiler() -> None:
    """Run profiler."""
    profiler = cProfile.Profile()
    profiler.enable()
    filename = os.path.join(
        os.environ["HOME"],
        "data",
        "TPTP-v8.1.2",
        "Problems",
        "GRP",
        "GRP001-1.p",
    )
    for _ in range(10):
        test_agent(["--problem_filename", filename])
    profiler.disable()
    profiler_report = io.StringIO()
    profiler_statistics = pstats.Stats(
        profiler, stream=profiler_report
    ).sort_stats(SortKey.TIME)
    profiler_statistics.print_stats(30)
    print(profiler_report.getvalue())


if __name__ == "__main__":
    run_profiler()
