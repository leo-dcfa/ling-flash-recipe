#!/usr/bin/env bash
# Build the image from the versions in pins.env.
# The tag names the engine commit, which is what the recipe will refer to.
# A label records this repo's commit, so an image can always be traced back
# to the Dockerfile and pins.env it came from.
#   ./build.sh                       extra arguments go to docker build,
#   ./build.sh --progress=plain      e.g. to see every line of output
set -euo pipefail
cd "$(dirname "$0")"

# The label would lie if the files it points to had uncommitted changes.
if ! git diff --quiet HEAD -- Dockerfile pins.env; then
  echo "build.sh: commit Dockerfile and pins.env first" >&2
  exit 1
fi

set -a; . ./pins.env; set +a

tag="ling-flash:sglang-${SGLANG_COMMIT:0:7}"
docker build \
  --build-arg SGLANG_REPO --build-arg SGLANG_COMMIT \
  --build-arg HUMMING_VERSION --build-arg HUMMING_SHA256 \
  --label org.opencontainers.image.source=https://github.com/leo-dcfa/ling-flash-recipe \
  --label org.opencontainers.image.revision="$(git rev-parse HEAD)" \
  -t "$tag" "$@" .
echo "build.sh: built $tag"
