FROM ubuntu:26.04

ARG RUNNER_VERSION=2.334.0
ARG RUNNER_SHA256=048024cd2c848eb6f14d5646d56c13a4def2ae7ee3ad12122bee960c56f3d271
ARG EMSDK_VERSION=4.0.22
ARG EMSDK_REF=15915cad554b707837024dc2758b6a1c5b94b036

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates clang-21 cmake curl g++ gcc git jq lld-21 \
        libblas-dev libicu-dev libjpeg-dev libncurses-dev libopenblas-dev libpcre2-dev \
        libpng-dev libreadline-dev libsqlite3-dev libssl-dev libwebp-dev libxml2-dev \
        libzstd-dev llvm-21 llvm-21-dev ninja-build nodejs pkg-config python3 \
        python3-numpy python3-venv tar unzip wget xz-utils zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/ld.lld-21 /usr/local/bin/ld.lld \
    && ln -s /usr/lib/llvm-21/bin/llvm-config /usr/local/bin/llvm-config

RUN python3 -m venv --system-site-packages /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --disable-pip-version-check \
        'PyYAML==6.0.2'

RUN mkdir -p /runner /runner-image \
    && curl --fail --location --retry 3 --output /runner-download.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && echo "${RUNNER_SHA256}  /runner-download.tar.gz" | sha256sum --check --strict \
    && tar --extract --gzip --file /runner-download.tar.gz --directory /runner-image \
    && cp --archive /runner-image/. /runner/ \
    && rm --force /runner-download.tar.gz \
    && /runner-image/bin/installdependencies.sh \
    && git clone --depth 1 https://github.com/emscripten-core/emsdk.git /opt/emsdk \
    && git -C /opt/emsdk fetch --depth 1 origin "${EMSDK_REF}" \
    && git -C /opt/emsdk checkout --detach "${EMSDK_REF}" \
    && /opt/emsdk/emsdk install "${EMSDK_VERSION}" \
    && /opt/emsdk/emsdk activate "${EMSDK_VERSION}" \
    && test -x /opt/emsdk/upstream/emscripten/emcc \
    && test -x /opt/emsdk/upstream/emscripten/em++ \
    && groupadd --gid 10001 runner \
    && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash runner \
    && mkdir -p /home/runner /work /deps \
    && chown --recursive runner:runner /home/runner /runner /runner-image /work /opt/emsdk

COPY --chmod=0755 runner-entrypoint.sh /usr/local/bin/runner-entrypoint
COPY --chmod=0755 toolchain-self-check.sh /usr/local/bin/eshkol-toolchain-self-check

ENV PATH=/opt/venv/bin:/opt/emsdk:/opt/emsdk/upstream/emscripten:/usr/lib/llvm-21/bin:/usr/local/bin:$PATH \
    HOME=/home/runner \
    ESHKOL_RUNNER_CONTAINER=1 \
    ESHKOL_RUNNER_EPHEMERAL=1 \
    FETCHCONTENT_BASE_DIR=/deps \
    LLVM_CONFIG_EXECUTABLE=/usr/lib/llvm-21/bin/llvm-config

WORKDIR /runner
USER runner
ENTRYPOINT ["/usr/local/bin/runner-entrypoint"]
