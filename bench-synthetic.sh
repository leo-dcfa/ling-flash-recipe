#!/usr/bin/env bash
# Measure decode speed the way the README's "Synthetic" column was measured:
# one request at a time, a 32K-token prompt, a 2K-token answer, greedy.
# It runs SGLang's own benchmark, from the server's image, against the running
# server, and appends one JSON line to results/synthetic.jsonl. The line holds
# the server's flags too, so it says what was measured.
#   ./bench-synthetic.sh baseline       the label is stored as "tag"
# Decode tok/s = 1000 / mean TPOT (ms). TPOT is the time per output token
# after the first, so the 32K prefill isn't in it.
set -euo pipefail
cd "$(dirname "$0")"
label="${1:?usage: ./bench-synthetic.sh <label>}"
set -a; . ./pins.env; set +a
image="ling-flash:sglang-${SGLANG_COMMIT:0:7}"
cache="${HF_HOME:-$HOME/.cache/huggingface}"
# hf is installed per user; a non-interactive shell may not have it on PATH.
PATH="$HOME/.local/bin:$PATH"

# Fetch the pinned prompts (once). The container then reads them offline.
data=ShareGPT_V3_unfiltered_cleaned_split.json
hf download --repo-type dataset "$SHAREGPT_REPO" "$data" \
  --revision "$SHAREGPT_REVISION" > /dev/null
# Where the HF cache keeps a revision, as the container sees it.
snapshot() { echo "/cache/huggingface/hub/$1--${2//\//--}/snapshots/$3"; }

# --network host: the server listens on the host's 127.0.0.1.
# --random-range-ratio 1: exactly 32K in and 2K out, not random lengths.
# bench_serving ignores EOS and sends temperature 0 unless told otherwise.
# --flush-cache: it warms up with the first prompt; without a flush the
# measured run would find that prompt cached.
# Two warnings are expected and harmless ("couldn't connect to huggingface.co",
# "use the Chat or Instruct model"): it looks for a chat template under the
# served name, ling-3.0-flash, which isn't a Hugging Face repo. The tokenizer
# that counts the tokens is the snapshot passed below.
mkdir -p results
docker run --rm --network host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
  -v "$cache:/cache/huggingface:ro" -v "$PWD/results:/results" \
  --entrypoint python3 "$image" -m sglang.benchmark.serving \
    --backend sglang --host 127.0.0.1 --port 8000 \
    --tokenizer "$(snapshot models "$MODEL_REPO" "$MODEL_REVISION")" \
    --dataset-name random \
    --dataset-path "$(snapshot datasets "$SHAREGPT_REPO" "$SHAREGPT_REVISION")/$data" \
    --random-input-len 32768 --random-output-len 2048 --random-range-ratio 1 \
    --num-prompts 3 --max-concurrency 1 --seed 1 --flush-cache \
    --output-file /results/synthetic.jsonl --tag "$label"

python3 - results/synthetic.jsonl <<'EOF'
import json, sys
r = json.loads(open(sys.argv[1]).readlines()[-1])
print(f"bench-synthetic.sh: {r['tag']}: decode {1000 / r['mean_tpot_ms']:.1f} tok/s,"
      f" TTFT {r['mean_ttft_ms'] / 1000:.1f} s, {r['completed']} prompts")
EOF
