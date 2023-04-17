FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as base

ENV HOME=/homedir \
    PYTHON_VERSION=3.9 \
    PATH=/opt/conda/envs/ai/bin:/opt/conda/bin:${PATH} \
    BITSANDBYTES_NOWELCOME=1

WORKDIR /app

RUN apt-get -y update && \
    apt-get install -y make git git-lfs curl wget unzip libaio-dev && \
    apt-get -y clean

# taken form pytorch's dockerfile
RUN curl -L -o ./miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm ./miniconda.sh

# create conda env
RUN conda create -n ai python=${PYTHON_VERSION} pip -y

FROM base as conda

# update conda
RUN conda update -n base -c defaults conda -y
# cmake
RUN conda install -c anaconda cmake -y

# necessary stuff
RUN pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 \
    transformers==4.27.4 \
    accelerate==0.18.0 \
    bitsandbytes==0.37.2 \
    aim==3.17.2 \
    peft==0.2.0 \
    pydantic \
    jsonlines \
    datasets \
    py-cpuinfo \
    pynvml \
    --no-cache-dir

# apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    git checkout 22.03 && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . && \
    cd .. && \
    rm -rf apex

# deepspeed
RUN git clone https://github.com/microsoft/DeepSpeed && \
    cd DeepSpeed && \
    git checkout v0.9.0 && \
    TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -v --global-option="build_ext" --global-option="-j8" --no-cache-dir . && \
    rm -rf DeepSpeed

# clean conda env
RUN conda clean -ya

RUN mkdir -p ~/.cache ~/.local && \
    chmod -R g+w /app ~/.cache ~/.local && \
    touch ~/.aim_profile && chmod g+w ~/.aim_profile && aim telemetry off
