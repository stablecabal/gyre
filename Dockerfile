ARG CUDA_VER=118
ARG CUDA_FULLVER=11.8.0


# ----- Build the core "dist image" base -----


FROM ghcr.io/stablecabal/gyre-devbase:pytorch112-cuda${CUDA_VER}-latest AS regularbase

# Install dependancies
ENV FLIT_ROOT_INSTALL=1
RUN /bin/micromamba -r /env -n gyre install -c defaults flit

# We copy only the minimum for flit to run so avoid cache invalidation on code changes
COPY pyproject.toml .
COPY gyre/__init__.py gyre/
RUN touch README.md
RUN /bin/micromamba -r /env -n gyre run flit install --pth-file
RUN /bin/micromamba -r /env -n gyre run pip cache purge

# Setup NVM & Node for Localtunnel
ENV NVM_DIR=/nvm
ENV NODE_VERSION=16.18.0

RUN mkdir -p $NVM_DIR

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default


# ----- Build MMCV -----

FROM regularbase AS mmcvbase
ARG MMCV_REPO=https://github.com/open-mmlab/mmcv.git
ARG MMCV_REF=v1.7.1
ARG MAX_JOBS=8
COPY docker_support/cuda_archs.sh /

WORKDIR /
RUN git clone $MMCV_REPO

WORKDIR /mmcv
RUN git checkout $MMCV_REF
RUN git submodule update --init --recursive
RUN /bin/micromamba -r /env -n gyre run pip install ninja psutil

ENV FORCE_CUDA=1
ENV MMCV_WITH_OPS=1
ENV MAX_JOBS=$MAX_JOBS
RUN TORCH_CUDA_ARCH_LIST="`/cuda_archs.sh`" /bin/micromamba -r /env -n gyre run pip install .

RUN tar cvjf /mmcv.tbz /env/envs/gyre/lib/python3.*/site-packages/mmcv*

# ----- Build bitsandbytes -----


FROM ghcr.io/stablecabal/gyre-devbase:pytorch112-cuda${CUDA_VER}-latest AS bitsandbytesbase
ARG BANDB_REF=main
ARG CUDA_VER

WORKDIR /
RUN git clone https://github.com/TimDettmers/bitsandbytes.git

WORKDIR /bitsandbytes
RUN git checkout $BANDB_REF

#COPY docker_support/bitsandbytes.sm89.diff /
#RUN patch -p1 < /bitsandbytes.sm89.diff 

ENV CUDA_VERSION=${CUDA_VER}
RUN /bin/micromamba -r /env -n gyre run make `echo ${CUDA_VER} | sed -e 's/118/cuda12x/' | sed -e 's/11./cuda11x/'`
RUN /bin/micromamba -r /env -n gyre run python setup.py bdist_wheel


# ----- Build triton -----


FROM ghcr.io/stablecabal/gyre-devbase:pytorch112-cuda${CUDA_VER}-latest AS tritonbase
ARG TRITON_REF=tags/v1.0

WORKDIR /
RUN git clone https://github.com/openai/triton.git

WORKDIR /triton
RUN git checkout $TRITON_REF

WORKDIR /triton/python
RUN /bin/micromamba -r /env -n gyre run pip install cmake
RUN apt install -y zlib1g-dev libtinfo-dev 
RUN /bin/micromamba -r /env -n gyre run pip install .

RUN tar cvjf /triton.tbz /env/envs/gyre/lib/python3.*/site-packages/triton*


# ----- Build xformers (on top of triton) -----


FROM tritonbase AS xformersbase
ARG XFORMERS_REPO=https://github.com/facebookresearch/xformers.git
ARG XFORMERS_REF=main
ARG MAX_JOBS=8
COPY docker_support/cuda_archs.sh /

WORKDIR /
RUN git clone $XFORMERS_REPO

WORKDIR /xformers
RUN git checkout $XFORMERS_REF
RUN git submodule update --init --recursive
RUN /bin/micromamba -r /env -n gyre run pip install -r requirements.txt
RUN /bin/micromamba -r /env -n gyre run pip install ninja

ENV FORCE_CUDA=1
ENV MAX_JOBS=$MAX_JOBS
RUN TORCH_CUDA_ARCH_LIST="`/cuda_archs.sh`" /bin/micromamba -r /env -n gyre run pip install .

RUN tar cvjf /xformers.tbz /env/envs/gyre/lib/python3.*/site-packages/xformers*


# ----- Build deepspeed (on top of triton) -----


FROM tritonbase AS deepspeedbase
ARG DEEPSPEED_REF=tags/v0.7.4
COPY docker_support/cuda_archs.sh /

RUN git clone https://github.com/microsoft/DeepSpeed.git

WORKDIR /

WORKDIR /DeepSpeed
RUN git checkout $DEEPSPEED_REF
RUN apt install -y libaio-dev

ENV DS_BUILD_OPS=1
ENV DS_BUILD_SPARSE_ATTN=0
RUN TORCH_CUDA_ARCH_LIST="`/cuda_archs.sh`" /bin/micromamba -r /env -n gyre run pip install .

RUN tar cvjf /deepspeed.tbz /env/envs/gyre/lib/python3.*/site-packages/deepspeed*


# ----- Build the basic inference server image -----


FROM nvidia/cuda:${CUDA_FULLVER}-cudnn8-runtime-ubuntu20.04 AS basic

COPY --from=regularbase /bin/micromamba /bin/
RUN mkdir -p /env/envs
COPY --from=regularbase /env/envs /env/envs/
RUN mkdir -p /nvm
COPY --from=regularbase /nvm /nvm/

COPY --from=mmcvbase /mmcv.tbz /
RUN tar xvjf /mmcv.tbz
COPY --from=mmcvbase /mmcv/requirements/runtime.txt /
RUN /bin/micromamba -r /env -n gyre run pip install -r runtime.txt
RUN rm runtime.txt


# Setup NVM & Node for Localtunnel
ENV NVM_DIR=/nvm
ENV NODE_VERSION=16.18.0

ENV NODE_PATH $NVM_DIR/versions/node/v$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

RUN npm install -g localtunnel

# Now we can copy everything we need
COPY nonfree /nonfree/
COPY gyre /gyre/
COPY server.py .

# Set up some config files
RUN mkdir -p /huggingface
RUN mkdir -p /weights
RUN mkdir -p /lora
RUN mkdir -p /embedding
RUN mkdir -p /config
COPY gyre/config/. /config/

# Set up some environment files

ENV HF_HOME=/huggingface
ENV HF_API_TOKEN=mustset
ENV SD_ENGINECFG=/config/engines.yaml
ENV SD_WEIGHT_ROOT=/weights
ENV SD_LOCAL_RESOURCE_1=embedding:/embedding
ENV SD_LOCAL_RESOURCE_2=lora:/lora


CMD [ "/bin/micromamba", "-r", "env", "-n", "gyre", "run", "python", "./server.py" ]


# ----- Build the basic inference server image + xformers -----


FROM basic as xformers

COPY --from=xformersbase /xformers/requirements.txt /
RUN /bin/micromamba -r /env -n gyre run pip install -r requirements.txt
RUN rm requirements.txt

COPY --from=tritonbase /triton.tbz /
RUN tar xvjf /triton.tbz
COPY --from=xformersbase /xformers.tbz /
RUN tar xvjf /xformers.tbz

RUN rm /*.tbz

CMD [ "/bin/micromamba", "-r", "env", "-n", "gyre", "run", "python", "./server.py" ]


# ----- Build the inference server image with training support -----
# (based on a -devel image instead of -runtime, but otherwise identical to basic)


FROM nvidia/cuda:${CUDA_FULLVER}-cudnn8-devel-ubuntu20.04 AS basic-training

COPY --from=regularbase /bin/micromamba /bin/
RUN mkdir -p /env/envs
COPY --from=regularbase /env/envs /env/envs/
RUN mkdir -p /nvm
COPY --from=regularbase /nvm /nvm/

COPY --from=mmcvbase /mmcv.tbz /
RUN tar xvjf /mmcv.tbz
COPY --from=mmcvbase /mmcv/requirements/runtime.txt /
RUN /bin/micromamba -r /env -n gyre run pip install -r runtime.txt
RUN rm runtime.txt

# Setup NVM & Node for Localtunnel
ENV NVM_DIR=/nvm
ENV NODE_VERSION=16.18.0

ENV NODE_PATH $NVM_DIR/versions/node/v$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

RUN npm install -g localtunnel

# Now we can copy everything we need
COPY nonfree /nonfree/
COPY gyre /gyre/
COPY server.py .

# Set up some config files
RUN mkdir -p /huggingface
RUN mkdir -p /weights
RUN mkdir -p /config
COPY gyre/config/. /config/

# Set up some environment files

ENV HF_HOME=/huggingface
ENV HF_API_TOKEN=mustset
ENV SD_ENGINECFG=/config/engines.yaml
ENV SD_WEIGHT_ROOT=/weights
ENV SD_LOCAL_RESOURCE_1=embedding:/embedding
ENV SD_LOCAL_RESOURCE_2=lora:/lora

CMD [ "/bin/micromamba", "-r", "env", "-n", "gyre", "run", "python", "./server.py" ]


# ----- Build the inference server image with training support + xformers, deepspeed, and bitsandbytes -----


FROM basic-training as xformers-training

COPY --from=xformersbase /xformers/requirements.txt /
RUN /bin/micromamba -r /env -n gyre run pip install -r requirements.txt
RUN rm requirements.txt

COPY --from=deepspeedbase /DeepSpeed/requirements/requirements.txt /
RUN /bin/micromamba -r /env -n gyre run pip install -r requirements.txt
RUN rm requirements.txt

COPY --from=tritonbase /triton.tbz /
RUN tar xvjf /triton.tbz
COPY --from=xformersbase /xformers.tbz /
RUN tar xvjf /xformers.tbz
COPY --from=deepspeedbase /deepspeed.tbz /
RUN tar xvjf /deepspeed.tbz

RUN rm /*.tbz

COPY --from=bitsandbytesbase /bitsandbytes/dist/*.whl /
RUN /bin/micromamba -r /env -n gyre run pip install /*.whl

RUN rm /*.whl

CMD [ "/bin/micromamba", "-r", "env", "-n", "gyre", "run", "python", "./server.py" ]