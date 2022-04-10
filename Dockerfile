FROM jupyter/base-notebook:python-3.9.7
USER root
COPY examples/*.ipynb /home/jovyan/
RUN chown -R jovyan /home/jovyan/*.ipynb
RUN apt-get update
RUN apt-get install -y unzip
USER jovyan
RUN wget https://github.com/vprover/vampire/releases/download/v4.6.1/vampire_z3_rel_static_master_5921.2.zip -O vampire.zip
RUN unzip vampire.zip
RUN mkdir -p $HOME/.local/bin/
RUN mv vampire_z3_rel_static_master_5921 $HOME/.local/bin/vampire
RUN chmod u+x $HOME/.local/bin/vampire
ENV PATH $HOME/.local/bin:${PATH}
RUN pip install gym-saturation
