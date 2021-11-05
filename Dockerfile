FROM jupyter/base-notebook:python-3.9.7
COPY examples/example.ipynb /home/jovyan
ENV PATH /home/jovyan/.local/bin:${PATH}
RUN pip install gym-saturation
