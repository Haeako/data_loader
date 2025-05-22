apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    cmake \
    build-essential \
    libopencv-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

TRT_VERSION=10.7.0.23
TRT_TARBALL=TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz
TRT_URL=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/${TRT_TARBALL}
wget --quiet --content-disposition TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz && \
    tar -xzf TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz && \
    rm TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz
TENSORRT_ROOT=/opt/TensorRT-${TRT_VERSION}
LD_LIBRARY_PATH=${TENSORRT_ROOT}/lib:$LD_LIBRARY_PATH
PATH=${TENSORRT_ROOT}/include
LIBRARY_PATH=${TENSORRT_ROOT}/lib:$LIBRARY_PATH
PATH=${TENSORRT_ROOT}/bin:$PATH

pip install --upgrade pip setuptools wheel
pip install --no-cache-dir torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu126

git clone --branch release/0.21 --depth 1 https://github.com/pytorch/vision.git /tmp/vision && \
    cd /tmp/vision && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_PREFIX_PATH="${CMAKE_PYTORCH_PATH}" && \
    cmake --build . --parallel $(nproc) && \
    cmake --install . --prefix=/opt/torchvision && \
    rm -rf /tmp/vision

WEIGHTS_FILE=co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth
WEIGHTS_FILE_PATH=/workspace/${WEIGHTS_FILE}
wget --quiet --content-disposition -O $WEIGHTS_FILE_PATH https://download.openmmlab.com/mmdetection/v3.0/codetr/${WEIGHTS_FILE}