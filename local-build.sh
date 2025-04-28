#!/bin/bash

set -e
PACKAGE_NAME=gym_saturation
cd doc
make clean html coverage
cat _build/coverage/python.txt
cd ..
ruff check
pylint ${PACKAGE_NAME} scripts
mypy ${PACKAGE_NAME} scripts
coverage run -m pytest
coverage report --show-missing --fail-under=100
pyroma -n 10 .
scc --no-cocomo --by-file -i py ${PACKAGE_NAME}
