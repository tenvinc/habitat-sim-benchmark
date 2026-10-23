# GPU habitat-sim (CUDA). CPU runs use the same image with gpu_device_id=-1.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libassimp-dev \
    libboost-all-dev \
    libbullet-dev \
    libegl1 \
    libegl1-mesa-dev \
    libgl1 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libglew-dev \
    libglm-dev \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libx11-6 \
    libxext6 \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

SHELL ["/bin/bash", "-lc"]

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

COPY environment.gpu.yml /workspace/environment.gpu.yml
WORKDIR /workspace

RUN conda env create -f environment.gpu.yml \
    && conda clean -afy

ARG HABITAT_BUILD_PARALLEL=2
ENV CMAKE_BUILD_PARALLEL_LEVEL=${HABITAT_BUILD_PARALLEL}

ARG HABITAT_SIM_REF=v0.3.3
RUN source activate habitat \
    && git clone --depth 1 --branch "${HABITAT_SIM_REF}" --recursive \
        https://github.com/facebookresearch/habitat-sim.git \
    && cd habitat-sim \
    && pip install --no-cache-dir -r requirements.txt \
    && python setup.py build_ext --parallel "${HABITAT_BUILD_PARALLEL}" \
        install --headless --with-cuda --bullet \
    && python -c "import habitat_sim; assert habitat_sim.cuda_enabled"

ARG HABITAT_LAB_REF=stable
RUN source activate habitat \
    && git clone --depth 1 --branch "${HABITAT_LAB_REF}" https://github.com/facebookresearch/habitat-lab.git \
    && cd habitat-lab \
    && pip install --no-cache-dir -e habitat-lab

COPY scripts/setup_mp3d_sample_data.sh /workspace/scripts/setup_mp3d_sample_data.sh
RUN chmod +x /workspace/scripts/setup_mp3d_sample_data.sh

ENV GLOG_minloglevel=2
ENV MAGNUM_LOG=quiet

WORKDIR /workspace
