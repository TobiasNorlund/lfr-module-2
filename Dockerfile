FROM conda/miniconda3:latest

COPY bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
RUN mkdir -p $DOCKER_WORKSPACE_PATH/src $DOCKER_WORKSPACE_PATH/.home
WORKDIR $DOCKER_WORKSPACE_PATH/src
ENV HOME=$DOCKER_WORKSPACE_PATH/.home

RUN apt update && apt install -y libsm6 libxext6 libxrender-dev

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
# Pull the environment name out of the environment.yml
RUN echo "\nsource activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" >> /etc/bash.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

