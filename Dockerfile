FROM conda/miniconda3:latest

COPY bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
RUN mkdir -p $DOCKER_WORKSPACE_PATH/src $DOCKER_WORKSPACE_PATH/.home
WORKDIR $DOCKER_WORKSPACE_PATH/src
ENV HOME=$DOCKER_WORKSPACE_PATH/.home

RUN apt update && apt install -y libsm6 libxext6 libxrender-dev

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
RUN conda install jupyter -y -n WASP_rprl_mdl2
# Pull the environment name out of the environment.yml
RUN echo "\nsource activate WASP_rprl_mdl2" >> /etc/bash.bashrc
ENV PATH "/opt/conda/envs/WASP_rprl_mdl2/bin:$PATH"

