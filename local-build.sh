#!/bin/bash

set -e
PACKAGE_NAME=gym_saturation
cd doc
make clean html
cd ..
pydocstyle ${PACKAGE_NAME}
pylint ${PACKAGE_NAME}
mypy ${PACKAGE_NAME}
pytest --cov-report term-missing
scc -i py ${PACKAGE_NAME}
