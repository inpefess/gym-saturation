#!/bin/bash

set -e
PACKAGE_NAME=gym_saturation
cd doc
make clean html
cd ..
flake8 ${PACKAGE_NAME}
pylint --rcfile=.pylintrc ${PACKAGE_NAME}
mypy --config-file mypy.ini ${PACKAGE_NAME}
pytest --cov-report term-missing
scc -i py ${PACKAGE_NAME}
