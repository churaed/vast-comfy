ARG CUDA_VERSION=12.1.0-devel

FROM --platform=amd64 docker.io/nvidia/cuda:${CUDA_VERSION}-ubuntu22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        ffmpeg \
        git \
        # TODO not sure all of this is required, remove unnnecessary
        libegl1 \
        libegl1-mesa-dev \
        libgl1 \
        libglib2.0-0 \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libgles2 \
        libgles2-mesa-dev \
        libglib2.0-0 \
        libglvnd-dev \
        libglvnd0 \
        libglx0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ninja-build \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        wget \
        && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python & \
    ln -s /usr/bin/python3.11 /usr/bin/python3 & \
    ln -s /usr/bin/pip3.11 /usr/bin/pip

RUN python -m pip install --upgrade pip

RUN adduser --uid 1001 -q user && \
    mkdir /app && chown user /app
USER user

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

WORKDIR /app

# Clone ComfyUI repository
RUN git clone "https://github.com/comfyanonymous/ComfyUI.git" ./ 

# Setup a virtual environment
RUN python -m venv venv

# Activate the virtual environment
ENV PATH="/app/venv/bin:$PATH" \
    VIRTUAL_ENV="/app/venv" \
    PYTHONPATH="/app/venv/lib/python3.11/site-packages" \
    PYTHONUSERBASE="/app/venv"

# Install the requirements
RUN pip install --no-cache -r requirements.txt

WORKDIR /app/custom_nodes/ComfyUI-3D-Pack/
COPY --chown=user:user requirements.txt ./
COPY --chown=user:user install.py ./
RUN pip install --no-cache -r requirements.txt \
    # post requirements installation require gpu, setup
    # `nvidia-container-runtime`, for docker, see
    # https://stackoverflow.com/a/61737404
    # those seem to be missed
    ninja rembg[gpu] open_clip_torch

COPY --chown=user:user ./ ./
RUN python install.py

# Install Custom Nodes
WORKDIR /app/custom_nodes/

# Essential nodes
RUN git clone "https://github.com/ltdrdata/ComfyUI-Impact-Pack" && \
    cd ComfyUI-Impact-Pack && \
    # git reset --hard ab17f8886945b0d36478950fe532164a8b569cc7 && \
    pip install --no-cache -r requirements.txt
RUN git clone "https://github.com/kijai/ComfyUI-KJNodes" && \
    cd ComfyUI-KJNodes && \
    # git reset --hard ffafc9c2c675ce4e89386c725def331a88274004 && \
    pip install --no-cache -r requirements.txt
RUN git clone "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite" && \
    cd ComfyUI-VideoHelperSuite && \
    # git reset --hard 70faa9bcef65932ab72e7404d6373fb300013a2e && \
    pip install --no-cache -r requirements.txt

# Extra nodes
RUN git clone "https://github.com/cubiq/ComfyUI_IPAdapter_plus" && \
    cd ComfyUI_IPAdapter_plus 
RUN git clone "https://github.com/ssitu/ComfyUI_UltimateSDUpscale" --recursive && \
    cd ComfyUI_UltimateSDUpscale
RUN git clone "https://github.com/ltdrdata/ComfyUI-Inspire-Pack" && \
    cd ComfyUI-Inspire-Pack && \
    pip install --no-cache -r requirements.txt
RUN git clone "https://github.com/edenartlab/eden_comfy_pipelines" && \
    cd eden_comfy_pipelines && \
    pip install --no-cache -r requirements.txt
RUN git clone "https://github.com/WASasquatch/was-node-suite-comfyui" && \
    cd was-node-suite-comfyui && \
    pip install --no-cache -r requirements.txt

# Clone comfyui-manager to handle extra nodes
RUN git clone "https://github.com/ltdrdata/ComfyUI-Manager.git" && \
    cd ComfyUI-Manager

WORKDIR /app
ENTRYPOINT [ "python", "main.py", "--listen", "0.0.0.0" ]
