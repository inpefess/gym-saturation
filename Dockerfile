FROM jupyter/base-notebook:python-3.9.7
USER root
COPY examples/*.ipynb /home/jovyan/
RUN chown -R jovyan /home/jovyan/*.ipynb
USER jovyan
ENV PATH /home/jovyan/.local/bin:${PATH}
RUN pip install gym-saturation
