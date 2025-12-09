# Eshkol Debian Debug Build
FROM debian:12-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    g++ \
    llvm-17-dev \
    libreadline-dev \
    file \
    gdb \
    && rm -rf /var/lib/apt/lists/*

# Copy source
WORKDIR /app
COPY . .

# Build with debug symbols
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=/usr

RUN cmake --build build --parallel

# Create package
WORKDIR /app/build
RUN cpack -G DEB

# Keep container running for file extraction
CMD ["sleep", "infinity"]
