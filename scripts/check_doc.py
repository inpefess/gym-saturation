# Copyright 2021-2023 Boris Shminke
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
Check Spelling in the Documentation
====================================

A simple script to check the ``rst`` files content.
"""
import os

from enchant import DictWithPWL
from enchant.checker import SpellChecker

doc_folder = os.path.join("doc", "source")
for dirpath, dirnames, filenames in os.walk(doc_folder):
    for filename in filenames:
        if os.path.splitext(filename)[1] == ".rst":
            checker = SpellChecker(DictWithPWL("en_GB", "spelling.dict"))
            with open(
                os.path.join(dirpath, filename), "r", encoding="utf-8"
            ) as doc_file:
                doc_text = doc_file.read()
            checker.set_text(doc_text)
            print(
                filename,
                {spelling_problem.word for spelling_problem in checker},
            )
