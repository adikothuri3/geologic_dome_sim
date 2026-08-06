#!/usr/bin/env bash
# Phase 3: install LingBot-Map into its own venv (~/venvs/lingbot).
#
# Separate venv on purpose: LingBot-Map pins torch 2.8/cu128, and ~/venvs/dome
# holds the jax 0.9.2 pin that brax needs. Mixing CUDA stacks in one venv is how
# the Phase 2 box gets broken. Same reasoning as the dimos venv -- see notes/decisions.md.
#
# WSL's network stalls fresh HTTPS connections roughly 1 in 5 (notes/setup.md),
# so every network call goes through retry().
set -uo pipefail

SRC=~/src/lingbot-map
VENV=~/venvs/lingbot
CKPT_DIR=~/ckpt/lingbot-map
LOG=/mnt/c/Users/Aditya/VSCode/GeologicDome/runs/lingbot_install.log

mkdir -p "$(dirname "$LOG")" "$CKPT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

retry() {
  local n=0 max=5
  until "$@"; do
    n=$((n + 1))
    if [ "$n" -ge "$max" ]; then
      log "FAILED after $max attempts: $*"
      return 1
    fi
    log "retry $n/$max: $*"
    sleep 5
  done
}

export GIT_HTTP_LOW_SPEED_LIMIT=1000
export GIT_HTTP_LOW_SPEED_TIME=20

log "=== stage 1: clone ==="
if [ ! -d "$SRC/.git" ]; then
  retry git clone --depth 1 https://github.com/Robbyant/lingbot-map.git "$SRC" || exit 1
else
  log "already cloned"
fi

log "=== stage 2: venv (python 3.10 per upstream README) ==="
if [ ! -x "$VENV/bin/python" ]; then
  retry uv venv --python 3.10 "$VENV" || exit 1
fi
export VIRTUAL_ENV="$VENV"

log "=== stage 3: torch 2.8.0 + cu128 ==="
retry uv pip install --python "$VENV/bin/python" \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128 || exit 1

log "=== stage 4: lingbot-map + vis extras ==="
retry uv pip install --python "$VENV/bin/python" -e "$SRC[vis]" || exit 1

# FlashInfer is "recommended" upstream but is a large build and we fall back to
# --use_sdpa anyway; skip it rather than risk a long failing compile. Non-fatal.
log "=== stage 5: optional extras (non-fatal) ==="
retry uv pip install --python "$VENV/bin/python" onnxruntime-gpu huggingface_hub opencv-python-headless \
  || log "WARN: optional extras failed -- sky masking may be unavailable"

log "=== stage 6: verify ==="
"$VENV/bin/python" - <<'PY' 2>&1 | tee -a "$LOG"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu", p.name, f"{p.total_memory/2**30:.1f} GiB")
import viser; print("viser", viser.__version__)
PY

log "=== done ==="
