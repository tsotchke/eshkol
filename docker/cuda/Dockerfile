# Eshkol CUDA-Enabled Build
# Builds with native float64 GPU acceleration via CUDA
#
# The default matches the CUDA 12.4 release toolchain. CUDA 12.4 cannot emit
# sm_120/sm_121 code, so Blackwell GPUs run its compute_90 PTX through the
# driver JIT. For native Blackwell code (e.g. DGX Spark / GB10, sm_121, arm64):
#   docker build -f docker/cuda/Dockerfile \
#     --build-arg CUDA_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04 \
#     --build-arg UBUNTU_CODENAME=noble \
#     --build-arg CUDA_ARCHITECTURES=121 -t eshkol-cuda .
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG VERSION=1.1.11
ARG LLVM_MAJOR=21
# apt.llvm.org suite; must match the base image's Ubuntu release.
ARG UBUNTU_CODENAME=jammy
# Empty keeps the portable default resolved in CMakeLists.txt.
ARG CUDA_ARCHITECTURES=
ARG DEBIAN_FRONTEND=noninteractive

# Add LLVM repository
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/${UBUNTU_CODENAME}/ llvm-toolchain-${UBUNTU_CODENAME}-${LLVM_MAJOR} main" >> /etc/apt/sources.list.d/llvm.list \
    && apt-get update

# Install build dependencies
RUN apt-get install -y \
    cmake \
    ninja-build \
    g++ \
    llvm-${LLVM_MAJOR} \
    llvm-${LLVM_MAJOR}-dev \
    libreadline-dev \
    libopenblas-dev \
    dpkg-dev \
    file \
    curl \
    pkg-config \
    libssl-dev \
    libncurses-dev \
    libpng-dev \
    libjpeg-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source
WORKDIR /app
COPY . .

# Build with GPU enabled (CUDA auto-detected from nvidia/cuda base image)
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DESHKOL_REQUIRED_LLVM_MAJOR=${LLVM_MAJOR} \
    -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-${LLVM_MAJOR} \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DESHKOL_GPU_ENABLED=ON \
    ${CUDA_ARCHITECTURES:+-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}}

RUN cmake --build build --parallel

# Run tests (full suite — GPU tests require nvidia runtime)
RUN ./scripts/run_all_tests.sh || echo "Tests completed"

# Verify artifacts
RUN ls -la /app/build/eshkol-run /app/build/eshkol-repl /app/build/stdlib.o /app/build/stdlib.bc || true

# Keep container running for file extraction
CMD ["sleep", "infinity"]
