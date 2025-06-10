FROM inpefess/python_with_provers:2025.03.10
COPY gym_saturation ./gym_saturation
COPY pyproject.toml poetry.toml poetry.lock README.rst ./doc/example.py .
RUN pip install -e .
RUN pip install jupyterlab jupytext
RUN jupytext-config set-default-viewer python nest_asyncio
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", \
   "--ServerApp.token=''", "--no-browser"]
