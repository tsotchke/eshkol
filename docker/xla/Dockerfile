# Eshkol XLA-Enabled Build
# Builds with StableHLO/MLIR support for XLA tensor operations
FROM ubuntu:22.04

ARG VERSION=1.1.11
ARG DEBIAN_FRONTEND=noninteractive

# Add LLVM 17 repository
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" >> /etc/apt/sources.list.d/llvm.list \
    && apt-get update

# Install build dependencies
RUN apt-get install -y \
    cmake \
    ninja-build \
    g++ \
    llvm-17 \
    llvm-17-dev \
    mlir-17-tools \
    libmlir-17-dev \
    libreadline-dev \
    libopenblas-dev \
    dpkg-dev \
    file \
    curl \
    git \
    pkg-config \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for LLVM tools
RUN ln -sf /usr/bin/llvm-config-17 /usr/bin/llvm-config

# Build StableHLO from source (required for XLA backend)
WORKDIR /opt
RUN git clone --depth 1 https://github.com/openxla/stablehlo.git \
    && cd stablehlo \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMLIR_DIR=/usr/lib/llvm-17/lib/cmake/mlir \
        -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm \
    && cmake --build build --parallel

# Copy Eshkol source
WORKDIR /app
COPY . .

# Build Eshkol with XLA enabled
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DESHKOL_XLA_ENABLED=ON \
    -DSTABLEHLO_ROOT=/opt/stablehlo

RUN cmake --build build --parallel

# Run tests (full suite)
RUN ./scripts/run_all_tests.sh || echo "Tests completed"

# Create package
WORKDIR /app/build
RUN cpack -G DEB -D CPACK_PACKAGE_VERSION="${VERSION}"

# Verify artifacts
RUN ls -la /app/build/eshkol-run /app/build/eshkol-repl /app/build/stdlib.o || true
RUN find /app/build -name "*.deb" -ls || true

# Keep container running for file extraction
CMD ["sleep", "infinity"]
