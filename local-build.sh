#!/bin/bash

set -e
PACKAGE_NAME=gym_saturation
cd doc
make clean html coverage
cat _build/coverage/python.txt
cd ..
pydocstyle ${PACKAGE_NAME} scripts
flake8 ${PACKAGE_NAME} scripts
pylint ${PACKAGE_NAME} scripts
mypy ${PACKAGE_NAME} scripts
pytest --cov-report term-missing
pyroma -n 10 .
find ${PACKAGE_NAME} -name "*.py" | xargs -I {} pyupgrade --py38-plus {}
bandit -r ${PACKAGE_NAME}
bandit -r scripts
scc --no-cocomo --by-file -i py ${PACKAGE_NAME}
