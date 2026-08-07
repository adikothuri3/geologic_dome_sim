#!/usr/bin/env bash
# Linux-side toolchain for the legacy MuJoCo track (+ recon/DimOS deps). Run inside WSL.
#
#     bash sims/mujoco/scripts/setup_wsl.sh --all
#     bash sims/mujoco/scripts/setup_wsl.sh --base --phase2      # just what Phase 2 needs
#
# Stages:
#   --base    apt libs, uv, venv, MuJoCo, Menagerie, offscreen-render probe   (Phase 1 parity)
#   --phase2  JAX-CUDA + MuJoCo Playground + MJX; verifies the G1 env loads   (locomotion policy)
#   --phase4  Open3D                                                          (point cloud -> terrain)
#   --phase6  DimOS                                                           (robot OS integration)
#   --all     every stage above
#
# Later stages are non-fatal: a DimOS failure must not cost you a working Phase 2 box.
# Only --base and --phase2 abort on error.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repo root (script lives in sims/mujoco/scripts/)
VENV="$HOME/venvs/dome"
DIMOS_VENV="$HOME/venvs/dimos"
SRC="$HOME/src"
MENAGERIE="$SRC/menagerie"
PLAYGROUND="$SRC/playground"

DO_BASE=0; DO_P2=0; DO_P4=0; DO_P6=0
[[ $# -eq 0 ]] && { echo "no stage selected; use --all or --base/--phase2/--phase4/--phase6"; exit 2; }
for a in "$@"; do
  case "$a" in
    --base) DO_BASE=1 ;;
    --phase2) DO_P2=1 ;;
    --phase4) DO_P4=1 ;;
    --phase6) DO_P6=1 ;;
    --all) DO_BASE=1; DO_P2=1; DO_P4=1; DO_P6=1 ;;
    *) echo "unknown flag: $a"; exit 2 ;;
  esac
done

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
warn() { printf '\033[33mWARN: %s\033[0m\n' "$1"; }

# Measured on this box: ~1 in 5 fresh HTTPS connections from WSL to GitHub hangs until it
# times out, while sustained transfers run fine at 6 MB/s. Without these, git sits on a dead
# socket for minutes; with them it gives up after 20s and retry() gets its next attempt.
export GIT_HTTP_LOW_SPEED_LIMIT=1000
export GIT_HTTP_LOW_SPEED_TIME=20

# Every network call below is wrapped so one stalled connection doesn't abort a 20-minute
# install. Note this only helps commands that actually return non-zero -- see the menagerie
# stage for why the failing command must itself be wrapped, not just its neighbour.
retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    [[ $n -ge 3 ]] && { warn "giving up after 3 attempts: $*"; return 1; }
    warn "attempt $n failed, retrying in 5s: $*"
    sleep 5
  done
}

# ----------------------------------------------------------------- base --
if [[ $DO_BASE -eq 1 ]]; then
  step 'base: system libraries'
  sudo apt-get update
  sudo apt-get install -y \
    build-essential git curl ca-certificates \
    python3.12-venv \
    libegl1 libgl1 libglx-mesa0 libosmesa6 libglfw3

  step 'base: uv + python 3.12 venv'
  # Before the check, not after: uv installs itself here and would otherwise be re-downloaded
  # on every re-run because it isn't on PATH yet.
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    # WSL's NAT drops the occasional TLS handshake; a transient `curl: (28)` here once
    # aborted the entire run at its first network call. Retry before giving up.
    ok=0
    for attempt in 1 2 3; do
      if curl -LsSf --connect-timeout 15 --max-time 120 --retry 2 \
           https://astral.sh/uv/install.sh -o /tmp/uv-install.sh; then
        sh /tmp/uv-install.sh && { ok=1; break; }
      fi
      warn "uv install attempt $attempt failed; retrying"
      sleep 5
    done
    [[ $ok -eq 1 ]] || { echo 'FATAL: could not install uv'; exit 1; }
  fi
  # `uv venv` errors out on an existing venv rather than no-opping, which broke re-runs after
  # a later stage failed. Reuse it if it's already there; this script must be resumable.
  if [[ -x "$VENV/bin/python" ]]; then
    echo "reusing existing venv at $VENV"
  else
    uv venv --python 3.12 "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"

  # imageio-ffmpeg (not mediapy) so video writing needs no system ffmpeg -- same code path
  # as the Windows box. See notes/decisions.md.
  retry uv pip install mujoco numpy imageio imageio-ffmpeg
  python -c "import mujoco; print('mujoco', mujoco.__version__)"

  step 'base: menagerie (sparse: unitree_g1 only)'
  mkdir -p "$SRC"

  # The Windows box already holds a sparse Menagerie clone that Phase 1 renders from. Copying
  # unitree_g1 out of it costs a few seconds and zero network. That matters here: roughly one
  # in five fresh HTTPS connections to GitHub from WSL hangs until timeout, and the lazy blob
  # fetch that `sparse-checkout set` triggers on a --filter=blob:none clone is exactly such a
  # connection -- it is what killed this stage before.
  WIN_MENAGERIE=/mnt/c/Users/Aditya/src/menagerie
  if [[ ! -e "$MENAGERIE/unitree_g1/g1.xml" && -e "$WIN_MENAGERIE/unitree_g1/g1.xml" ]]; then
    echo "seeding from the Windows clone at $WIN_MENAGERIE (no network)"
    rm -rf "$MENAGERIE"
    mkdir -p "$MENAGERIE"
    cp -r "$WIN_MENAGERIE/unitree_g1" "$MENAGERIE/"
  fi

  if [[ ! -e "$MENAGERIE/unitree_g1/g1.xml" ]]; then
    # No local copy to seed from: clone for real. Both the clone and the blob fetch are
    # retried, since either can draw a stalled connection.
    [[ -d "$MENAGERIE/.git" ]] || retry git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie "$MENAGERIE"
    retry git -C "$MENAGERIE" sparse-checkout set unitree_g1
  fi
  ls "$MENAGERIE/unitree_g1/g1.xml" >/dev/null

  # sims/mujoco/xmls/scene_g1_hfield.xml includes menagerie/unitree_g1/g1.xml relative to itself.
  MJ_MENAGERIE="$REPO/sims/mujoco/xmls/menagerie"
  if [[ ! -e "$MJ_MENAGERIE/unitree_g1/g1.xml" ]]; then
    rm -rf "$MJ_MENAGERIE"
    ln -sfn "$MENAGERIE" "$MJ_MENAGERIE" 2>/dev/null || true
    if [[ ! -e "$MJ_MENAGERIE/unitree_g1/g1.xml" ]]; then
      warn 'symlink unavailable on this filesystem; copying instead'
      rm -rf "$MJ_MENAGERIE"; mkdir -p "$MJ_MENAGERIE"
      cp -r "$MENAGERIE/unitree_g1" "$MJ_MENAGERIE/"
    fi
  fi
  echo "menagerie ok"

  step 'base: offscreen render backend'
  grep -q 'MUJOCO_GL' "$HOME/.bashrc" || echo 'export MUJOCO_GL=egl' >> "$HOME/.bashrc"
  grep -q 'venvs/dome' "$HOME/.bashrc" || echo "source $VENV/bin/activate" >> "$HOME/.bashrc"
  export MENAGERIE_DIR="$MENAGERIE"
  render_ok=0
  for backend in egl osmesa; do
    echo "--- trying MUJOCO_GL=$backend"
    if MUJOCO_GL="$backend" python "$REPO/sims/mujoco/scripts/check_render.py"; then
      sed -i "s/^export MUJOCO_GL=.*/export MUJOCO_GL=$backend/" "$HOME/.bashrc"
      echo "rendering works with MUJOCO_GL=$backend"
      render_ok=1; break
    fi
    if [[ "$backend" == egl && ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
      echo 'installing NVIDIA EGL ICD so glvnd finds the WSL driver'
      sudo mkdir -p /usr/share/glvnd/egl_vendor.d
      echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null
      if MUJOCO_GL=egl python "$REPO/sims/mujoco/scripts/check_render.py"; then
        sed -i 's/^export MUJOCO_GL=.*/export MUJOCO_GL=egl/' "$HOME/.bashrc"
        echo 'rendering works with MUJOCO_GL=egl (after ICD fix)'
        render_ok=1; break
      fi
    fi
  done
  [[ $render_ok -eq 1 ]] || { echo 'FATAL: no working GL backend'; exit 1; }
fi

# --------------------------------------------------------------- phase2 --
if [[ $DO_P2 -eq 1 ]]; then
  export PATH="$HOME/.local/bin:$PATH"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"

  step 'phase2: JAX with CUDA 12 (pinned -- see below)'
  # Pinned, and it must stay pinned. brax 0.14.2 (the newest release, and what Playground
  # requires) still calls jax.device_put_replicated in ppo/train.py:756. jax deprecated that
  # in 0.8.1 and REMOVED it in 0.10.0 with the pmap -> jit(shard_map) migration, so an
  # unpinned `jax[cuda12]` resolves to 0.11.0 and every PPO run dies at startup with
  # `AttributeError: jax.device_put_replicated is deprecated`. brax declares only
  # `jax>=0.4.6`, so nothing upstream prevents this.
  # 0.9.2 is the last release with the API. Revisit when brax ships a fix; verify with
  # sims/mujoco/scripts/train_g1.py --smoke before unpinning.
  retry uv pip install "jax[cuda12]==0.9.2"
  python - <<'PY'
import jax
devs = jax.devices()
print("jax", jax.__version__, "devices:", devs)
if not any(d.platform == "gpu" for d in devs):
    raise SystemExit("FATAL: JAX sees no GPU. Check nvidia-smi inside WSL and the driver passthrough.")
PY

  step 'phase2: MuJoCo Playground (editable clone -- we need learning/train_jax_ppo.py)'
  if [[ ! -d "$PLAYGROUND/.git" ]]; then
    retry git clone --depth 1 https://github.com/google-deepmind/mujoco_playground "$PLAYGROUND"
  fi
  retry uv pip install -e "$PLAYGROUND"
  retry uv pip install mujoco-mjx

  step 'phase2: verify the G1 joystick env loads'
  python "$REPO/sims/mujoco/scripts/check_phase2.py"
fi

# --------------------------------------------------------------- phase4 --
if [[ $DO_P4 -eq 1 ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  step 'phase4: Open3D (point-cloud cleanup -> terrain)'
  if uv pip install open3d; then
    python -c "import open3d; print('open3d', open3d.__version__)" || warn 'open3d imported badly'
  else
    warn 'open3d install failed -- Phase 4 only, does not affect Phase 2'
  fi
fi

# --------------------------------------------------------------- phase6 --
if [[ $DO_P6 -eq 1 ]]; then
  export PATH="$HOME/.local/bin:$PATH"
  step 'phase6: DimOS (isolated venv)'
  # dimos[base] pulls pyaudio, which has no wheel and compiles against Python.h + portaudio.
  # Kept in this stage rather than --base: nothing before Phase 6 needs a compiler toolchain
  # for audio.
  sudo apt-get install -y python3.12-dev portaudio19-dev || \
    warn 'could not install pyaudio build deps'
  # DimOS resolves ~289 packages including torch and a second full CUDA stack, and it pins
  # numpy. Installing that into the Phase 2 venv would let a Phase 6 dependency silently
  # re-resolve numpy/jax under MJX -- trading a working locomotion policy for a robot-OS
  # integration that isn't due until late September. It gets its own venv.
  if [[ -x "$DIMOS_VENV/bin/python" ]]; then
    echo "reusing existing venv at $DIMOS_VENV"
  else
    uv venv --python 3.12 "$DIMOS_VENV"
  fi
  if retry uv pip install --python "$DIMOS_VENV/bin/python" 'dimos[base,unitree]'; then
    "$DIMOS_VENV/bin/python" -c "import dimos; print('dimos ok')" || warn 'dimos imported badly'
  else
    warn 'dimos install failed -- Phase 6 only, does not affect Phase 2. Revisit closer to Sept.'
  fi
fi

step 'done'
cat <<EOF
Environment ready. From $REPO with the venv active:

  Phase 1 (should still pass here):
    python sims/mujoco/scripts/inspect_model.py
    python sims/mujoco/terrain/drop_test.py
    python sims/mujoco/scripts/pose_and_render.py

  Phase 2:
    python sims/mujoco/scripts/check_phase2.py          # jax GPU + G1 env smoke test
    python sims/mujoco/scripts/train_g1.py --smoke      # ~2 min, proves the training loop runs
    python sims/mujoco/scripts/train_g1.py              # the real baseline run
EOF
