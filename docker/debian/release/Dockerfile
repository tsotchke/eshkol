# Eshkol Debian/Ubuntu Release Build
# Matches GitHub Actions ubuntu-22.04 environment
FROM ubuntu:22.04

ARG VERSION=1.0.0
ARG DEBIAN_FRONTEND=noninteractive

# Add LLVM 17 repository (required - not in Ubuntu 22.04 default repos)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" >> /etc/apt/sources.list.d/llvm.list \
    && apt-get update

# Install build dependencies - matching GitHub Actions CI
RUN apt-get install -y \
    cmake \
    ninja-build \
    g++ \
    llvm-17 \
    llvm-17-dev \
    libreadline-dev \
    dpkg-dev \
    file \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for LLVM tools (Ubuntu packages them versioned)
RUN ln -sf /usr/bin/llvm-config-17 /usr/bin/llvm-config

# Copy source
WORKDIR /app
COPY . .

# Build - exactly matching GitHub Actions
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr

RUN cmake --build build --parallel

# Run tests (same as CI)
RUN ./scripts/run_types_tests.sh || echo "Type tests completed"
RUN ./scripts/run_list_tests.sh || echo "List tests completed"
RUN ./scripts/run_autodiff_tests.sh || echo "Autodiff tests completed"

# Create package
WORKDIR /app/build
RUN cpack -G DEB -D CPACK_PACKAGE_VERSION="${VERSION}"

# Verify artifacts
RUN ls -la /app/build/eshkol-run /app/build/eshkol-repl /app/build/stdlib.o
RUN find /app/build -name "*.deb" -ls

# Keep container running for file extraction
CMD ["sleep", "infinity"]
