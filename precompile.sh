#!/usr/bin/env bash
# Compile FlashInfer's MXFP4 MoE kernel (fused_moe_120) before the baseline's
# first launch, with the memory to itself. The cookbook's step 4 does the
# same: compiling it while SGLang loads 60 GB of weights can run the box out
# of memory. The server later asks FlashInfer for the same module, finds it
# built, and skips the compile. Only the baseline uses this kernel.
#   ./precompile.sh        about 6-8 minutes the first time, seconds after
set -euo pipefail
cd "$(dirname "$0")"

set -a; . ./pins.env; set +a
image="ling-flash:sglang-${SGLANG_COMMIT:0:7}"

# The HF cache, mounted where sparkrun mounts it. FLASHINFER_WORKSPACE_BASE
# must match the recipe's, or the server won't find the kernel.
cache="${HF_HOME:-$HOME/.cache/huggingface}"

# The cookbook's rule: 10 GB of free memory per compile job, at most 8.
avail_gb=$(awk '/^MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
jobs=$(( avail_gb / 10 ))
(( jobs > 8 )) && jobs=8
(( jobs < 1 )) && jobs=1
echo "precompile.sh: ${avail_gb} GB available, ${jobs} compile jobs"

# The same call the server makes: SM 12.1 asks for module "121".
# Run as you, like sparkrun does, so the server can use what's built here.
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$cache:/cache/huggingface" \
  -e FLASHINFER_WORKSPACE_BASE=/cache/huggingface/ling-flash \
  -e MAX_JOBS="$jobs" \
  --entrypoint python3 "$image" -c '
import torch
from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
major, minor = torch.cuda.get_device_capability()
get_cutlass_fused_moe_module(f"{major * 10 + minor}")
print(f"precompile.sh: fused MoE kernel ready for SM {major}.{minor}")
'
