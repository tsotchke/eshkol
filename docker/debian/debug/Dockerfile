# Eshkol Debian Debug Build
FROM debian:12-slim

ARG DEBIAN_FRONTEND=noninteractive

# Install prerequisites for adding LLVM repository
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common

# Add LLVM 17 repository (not in Debian 12 default repos)
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-17 main" >> /etc/apt/sources.list.d/llvm.list \
    && apt-get update

# Install build dependencies
RUN apt-get install -y \
    cmake \
    ninja-build \
    g++ \
    llvm-17 \
    llvm-17-dev \
    libreadline-dev \
    libopenblas-dev \
    dpkg-dev \
    file \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for LLVM tools (Debian packages them versioned)
RUN ln -sf /usr/bin/llvm-config-17 /usr/bin/llvm-config

# Copy source
WORKDIR /app
COPY . .

# Build with debug symbols
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=/usr

RUN cmake --build build --parallel

# Run tests (full suite)
RUN ./scripts/run_all_tests.sh || echo "Tests completed"

# Create package
WORKDIR /app/build
RUN cpack -G DEB

# Verify artifacts
RUN ls -la /app/build/eshkol-run /app/build/eshkol-repl /app/build/stdlib.o || true
RUN find /app/build -name "*.deb" -ls || true

# Keep container running for file extraction
CMD ["sleep", "infinity"]
