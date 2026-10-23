# The Triton AMD CI image provides TheRock ROCm, Mirage, and rocjitsu.
# Its tag identifies a development build, so the digest is pinned.
ARG BASE=tritonci/triton-amd-ci:rocm10.2.0a20261012-pytorch2.16.0a0-gfx942-gfx950-gfx1250-r1@sha256:f717d27c0400b8ef7dad16d2d38ae56c0fe2153608af7ee0e0e98d80b98f3bba
FROM ${BASE}

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git \
      ninja-build \
 && rm -rf /var/lib/apt/lists/*

COPY rjsim /usr/local/bin/rjsim
WORKDIR /work
