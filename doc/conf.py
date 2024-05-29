# type: ignore
# Copyright 2021-2024 Boris Shminke
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

"""Sphinx config."""
# pylint: disable-all
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
project = "gym-saturation"
version = "0.11.8"
copyright = "2021-2024, Boris Shminke"
author = "Boris Shminke"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    # uncomment to rebuild examples
    # "sphinx_gallery.gen_gallery",
]
html_theme = "furo"
html_title = "gym-saturation documentation"
sphinx_gallery_conf = {
    "download_all_examples": False,
    "run_stale_examples": True,
    "image_scrapers": (),
    "reset_modules": (),
}
