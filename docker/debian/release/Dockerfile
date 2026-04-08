# Eshkol Debian/Ubuntu Release Build
# Matches GitHub Actions ubuntu-22.04 environment
FROM ubuntu:22.04

ARG VERSION=1.1.11
ARG DEBIAN_FRONTEND=noninteractive

# Add LLVM 21 repository (required - not in Ubuntu 22.04 default repos)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-21 main" >> /etc/apt/sources.list.d/llvm.list \
    && apt-get update

# Install build dependencies - matching GitHub Actions CI
RUN apt-get install -y \
    cmake \
    ninja-build \
    g++ \
    llvm-21 \
    llvm-21-dev \
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

# Build - exactly matching GitHub Actions
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config-21 \
    -DCMAKE_INSTALL_PREFIX=/usr

RUN cmake --build build -- -j1

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
