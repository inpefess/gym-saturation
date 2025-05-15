#!/bin/bash

set -e
PACKAGE_NAME=gym_saturation
cd doc
make clean html coverage
cat _build/coverage/python.txt
cd ..
ruff check
mypy ${PACKAGE_NAME}
pydoclint ${PACKAGE_NAME}
coverage run -m pytest
coverage report --show-missing --fail-under=100
pyroma -n 10 .
scc --no-cocomo --by-file -i py ${PACKAGE_NAME}
