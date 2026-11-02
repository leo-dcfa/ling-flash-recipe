#!/usr/bin/env bash
# One measurement, the same way every time: launch a recipe on this box, wait
# until it answers, ask it one question, run bench-synthetic.sh, stop it.
#   ./measure.sh <label> [recipe]       recipe defaults to ./ling-flash.yaml
# Two models loading at once can run a GB10 out of memory and hang it, which
# nearly happened when this recipe met the GLM lane. So it refuses to start
# while any other container runs here, and while the server loads it kills
# the server if free memory falls below 6 GB. The server's log is kept in
# ~/ling-flash-logs, because stopping the server deletes its container.
set -euo pipefail
cd "$(dirname "$0")"
label="${1:?usage: ./measure.sh <label> [recipe]}"
recipe="${2:-./ling-flash.yaml}"
# sparkrun is installed per user; a non-interactive shell may not have it.
PATH="$HOME/.local/bin:$PATH"

if [ -n "$(docker ps -q)" ]; then
  echo "measure.sh: free this box first; running now:" >&2
  docker ps --format '  {{.Names}}' >&2
  exit 1
fi

sparkrun run "$recipe" --hosts localhost --no-follow
# From here on, however the script ends, the server is stopped.
trap 'sparkrun stop "$recipe" --hosts localhost > /dev/null 2>&1 || true' EXIT
server=$(docker ps --filter name=sparkrun_ --format '{{.Names}}' | head -n 1)
mkdir -p ~/ling-flash-logs
log=~/ling-flash-logs/$(date +%Y%m%d-%H%M)-${label//[^A-Za-z0-9._-]/_}.log
docker exec "$server" tail -n +1 -F /tmp/sparkrun_serve.log > "$log" 2> /dev/null &

# Up to 40 minutes: a first Humming launch also compiles its kernels.
for _ in $(seq 2400); do
  if curl -sf -m 2 127.0.0.1:8000/v1/models > /dev/null; then break; fi
  avail_kb=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
  if (( avail_kb < 6 * 1024 * 1024 )); then
    docker kill "$server" > /dev/null
    echo "measure.sh: only $((avail_kb / 1024)) MB free, server killed; log: $log" >&2
    exit 1
  fi
  if [ -z "$(docker ps -q --filter "name=^$server\$")" ]; then
    echo "measure.sh: the server exited; log: $log" >&2
    exit 1
  fi
  sleep 1
done
curl -sf -m 2 127.0.0.1:8000/v1/models > /dev/null \
  || { echo "measure.sh: not ready after 40 minutes; log: $log" >&2; exit 1; }
echo "measure.sh: ready; server log: $log"

# A speedup that breaks the answers isn't one. One question with a known
# answer (3 hours 35 minutes) catches the obvious breakage, not subtle drift.
python3 - <<'EOF'
import json, urllib.request
body = {"model": "ling-3.0-flash", "temperature": 0, "max_tokens": 2048,
        "messages": [{"role": "user", "content":
                      "A train leaves at 09:40 and arrives at 13:15. How long is the trip?"}]}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
answer = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
print("measure.sh: smoke answer:", " ".join((answer or "").split())[:300])
EOF

./bench-synthetic.sh "$label"
