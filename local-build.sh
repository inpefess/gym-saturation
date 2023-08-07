#!/bin/bash

set -e
PACKAGE_NAME=gym_saturation
cd doc
make clean html
cd ..
pydocstyle ${PACKAGE_NAME} scripts
flake8 ${PACKAGE_NAME} scripts
pylint ${PACKAGE_NAME} scripts
mypy ${PACKAGE_NAME} scripts
pytest --cov-report term-missing
pyroma .
bandit -r ${PACKAGE_NAME}
bandit -r scripts
scc -i py ${PACKAGE_NAME}
