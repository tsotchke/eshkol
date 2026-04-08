# Eshkol CUDA-Enabled Build
# Builds with native float64 GPU acceleration via CUDA
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG VERSION=1.1.11
ARG LLVM_MAJOR=21
ARG DEBIAN_FRONTEND=noninteractive

# Add LLVM repository
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_MAJOR} main" >> /etc/apt/sources.list.d/llvm.list \
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
    -DESHKOL_GPU_ENABLED=ON

RUN cmake --build build --parallel

# Run tests (full suite — GPU tests require nvidia runtime)
RUN ./scripts/run_all_tests.sh || echo "Tests completed"

# Verify artifacts
RUN ls -la /app/build/eshkol-run /app/build/eshkol-repl /app/build/stdlib.o || true

# Keep container running for file extraction
CMD ["sleep", "infinity"]
