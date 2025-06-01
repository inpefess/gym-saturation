FROM inpefess/python_with_provers:2025.03.10
COPY gym_saturation ./gym_saturation
COPY pyproject.toml .
COPY poetry.toml .
COPY poetry.lock .
COPY README.rst .
RUN pip install -e .
RUN pip install jupyterlab
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", \
   "--ServerApp.token=''", "--no-browser"]
