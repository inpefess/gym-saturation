FROM inpefess/python_with_provers
USER root
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}
RUN addgroup --gid ${NB_UID} ${NB_USER}
RUN adduser --uid ${NB_UID} --gid ${NB_UID} ${NB_USER}
COPY examples/*.ipynb ${HOME}/
COPY gym_saturation ${HOME}/gym_saturation
COPY pyproject.toml ${HOME}
COPY poetry.toml ${HOME}
COPY poetry.lock ${HOME}
COPY README.rst ${HOME}
RUN apt-get update
RUN apt-get install -y unzip
WORKDIR ${HOME}
RUN pip install -U poetry
RUN poetry install
RUN chown -R ${NB_USER}:${NB_USER} ${HOME}
USER ${NB_USER}
RUN wget https://github.com/vprover/vampire/releases/download/v4.7/vampire4.7.zip -O vampire.zip
RUN unzip vampire.zip
RUN mkdir -p ${HOME}/.local/bin/
RUN mv vampire_z3_rel_static_HEAD_6295 ${HOME}/.local/bin/vampire
RUN chmod u+x ${HOME}/.local/bin/vampire
ENV PATH ${PYENV_ROOT}/versions/3.10.7/bin:${HOME}/.local/bin:${PATH}
ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", \
    "--ServerApp.token=passwd007", "--no-browser"]
