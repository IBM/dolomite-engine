# 1) Use the PyTorch base image with CUDA 12.4 and cuDNN 9
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# 2) Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3) Update system packages and install tools needed for building from source
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 4) Upgrade pip/setuptools
RUN pip install --upgrade pip setuptools wheel

# 5) (Optional) Reinstall PyTorch nightly for CUDA 12.x
#    If you trust the base image, you can remove or comment out the line below.
RUN pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# 7) Install dependencies inside the dolomite environment
RUN pip install --no-cache-dir \
    transformers \
    dataset \
    pydantic \
    safetensors

RUN pip install --no-cache-dir flash-attn
RUN pip install --no-cache-dir mamba-ssm[causal-conv1d]

# 8) Additional tools
RUN pip install wandb aim colorlog torchao
RUN pip install git+https://github.com/mayank31398/cute-kernels
RUN pip install git+https://github.com/shawntan/stickbreaking-attention

# 9) Install Dolomite Engine
RUN git clone --recursive https://github.com/NexaAI/dolomite-engine.git
RUN cd dolomite-engine && pip install -e .

# 10) By default, launch an interactive shell
CMD ["/bin/bash"]
