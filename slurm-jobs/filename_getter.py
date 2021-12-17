"""
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
"""

import os
from glob import glob


def get_filename() -> str:
    """
    :returns: a TPTP v7.5.0 CNF problem filename with the absolute path
        corresponding to an environment variable ``SLURM_ARRAY_TASK_ID``
        (automatically created by `Slurm's job array functionality <https://slurm.schedmd.com/job_array.html>`__)
    """
    return glob(
        os.path.join(
            os.environ["WORK"], "data", "TPTP-v7.5.0", "Problems", "*", "*-*.p"
        )
    )[int(os.environ["SLURM_ARRAY_TASK_ID"])]


if __name__ == "__main__":
    print(get_filename())
