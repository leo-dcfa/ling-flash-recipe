# syntax=docker/dockerfile:1
# Ling-3.0-flash on one GB10: inclusionAI's SGLang fork, installed the way the
# cookbook does it (uv, Python 3.11, CUDA 13), inside an image sparkrun can run.
# Versions come from pins.env:
#   set -a; . ./pins.env; set +a
#   docker build --build-arg SGLANG_REPO --build-arg SGLANG_COMMIT -t ling-flash:dev .

# CUDA 13.0.3: the CUDA that torch 2.13.0 ships with, and no newer than the
# host driver supports (13.0). "devel" adds nvcc, which FlashInfer uses to
# compile the baseline's MoE kernel on first launch. Pinned by digest.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266

# uv, copied from its official image rather than piped from an install script.
COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /usr/local/bin/uv

# git to fetch the fork, a compiler for the few packages built from source,
# procps because sparkrun checks on the server with ps.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates git procps \
 && rm -rf /var/lib/apt/lists/*

ARG SGLANG_REPO
ARG SGLANG_COMMIT

# Fetch exactly one commit. git checks the content against the hash. Then
# .git goes, so the version falls back to 0.0.0.dev0 (the pyproject allows it).
RUN : "${SGLANG_COMMIT:?pass --build-arg SGLANG_COMMIT from pins.env}" \
 && git init -q /src/sglang \
 && cd /src/sglang \
 && git fetch -q --depth 1 "$SGLANG_REPO" "$SGLANG_COMMIT" \
 && git checkout -q FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "$SGLANG_COMMIT" \
 && rm -rf .git

# Python 3.11 in a venv, as in the cookbook, but without its [all] extra:
# that adds image and video generation, which Ling doesn't use.
# The pyproject pins torch, FlashInfer and Humming but leaves many packages
# open. Resolve those as of 2026-09-08, the day after NVIDIA's run.
# No Rust extensions, as in the cookbook: local serving doesn't use them.
ENV UV_PYTHON_INSTALL_DIR=/opt/python VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN uv venv -q --python 3.11.15 /opt/venv \
 && UV_EXCLUDE_NEWER=2026-09-08T00:00:00Z SGLANG_BUILD_RUST_EXTS=none \
    MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 \
    uv pip install --no-cache /src/sglang/python
