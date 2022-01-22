#!/bin/bash

set -e
PACKAGE_NAME=gym_saturation
cd doc
make clean html
cd ..
pycodestyle --max-doc-length 10000 --ignore E203,E501,W503 ${PACKAGE_NAME}
pylint --rcfile=.pylintrc ${PACKAGE_NAME}
mypy --config-file mypy.ini ${PACKAGE_NAME}
pytest --cov ${PACKAGE_NAME} --cov-report term-missing --cov-fail-under=99 \
       --junit-xml test-results/saturation_gyn.xml ${PACKAGE_NAME}
scc -i py ${PACKAGE_NAME}
