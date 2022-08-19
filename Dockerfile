FROM jupyter/base-notebook:python-3.9.7
USER root
COPY examples/*.ipynb ${HOME}/
COPY gym_saturation ${HOME}/gym_saturation
COPY pyproject.toml ${HOME}
COPY poetry.toml ${HOME}
COPY poetry.lock ${HOME}
COPY README.rst ${HOME}
RUN chown -R ${NB_USER} ${HOME}/*.ipynb
RUN apt-get update
RUN apt-get install -y unzip
USER ${NB_USER}
RUN wget https://github.com/vprover/vampire/releases/download/v4.7/vampire4.7.zip -O vampire.zip
RUN unzip vampire.zip
RUN mkdir -p ${HOME}/.local/bin/
RUN mv vampire_z3_rel_static_HEAD_6295 ${HOME}/.local/bin/vampire
RUN chmod u+x ${HOME}/.local/bin/vampire
ENV PATH ${HOME}/.local/bin:${PATH}
RUN pip install poetry
RUN poetry install
