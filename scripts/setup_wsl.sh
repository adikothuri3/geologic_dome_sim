#!/usr/bin/env bash
# Linux-side toolchain for the whole pipeline. Run inside WSL.
#
#     bash scripts/setup_wsl.sh --all
#     bash scripts/setup_wsl.sh --base --phase2      # just what Phase 2 needs
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

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/venvs/dome"
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

# ----------------------------------------------------------------- base --
if [[ $DO_BASE -eq 1 ]]; then
  step 'base: system libraries'
  sudo apt-get update
  sudo apt-get install -y \
    build-essential git curl ca-certificates \
    libegl1 libgl1 libglx-mesa0 libosmesa6 libglfw3

  step 'base: uv + python 3.12 venv'
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"

  # imageio-ffmpeg (not mediapy) so video writing needs no system ffmpeg -- same code path
  # as the Windows box. See notes/decisions.md.
  uv pip install mujoco numpy imageio imageio-ffmpeg
  python -c "import mujoco; print('mujoco', mujoco.__version__)"

  step 'base: menagerie (sparse: unitree_g1 only)'
  mkdir -p "$SRC"
  if [[ ! -d "$MENAGERIE/.git" ]]; then
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie "$MENAGERIE"
  fi
  git -C "$MENAGERIE" sparse-checkout set unitree_g1
  ls "$MENAGERIE/unitree_g1/g1.xml" >/dev/null

  # sim/scene_g1_hfield.xml includes menagerie/unitree_g1/g1.xml relative to itself.
  if [[ ! -e "$REPO/sim/menagerie/unitree_g1/g1.xml" ]]; then
    rm -rf "$REPO/sim/menagerie"
    ln -sfn "$MENAGERIE" "$REPO/sim/menagerie" 2>/dev/null || true
    if [[ ! -e "$REPO/sim/menagerie/unitree_g1/g1.xml" ]]; then
      warn 'symlink unavailable on this filesystem; copying instead'
      rm -rf "$REPO/sim/menagerie"; mkdir -p "$REPO/sim/menagerie"
      cp -r "$MENAGERIE/unitree_g1" "$REPO/sim/menagerie/"
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
    if MUJOCO_GL="$backend" python "$REPO/scripts/check_render.py"; then
      sed -i "s/^export MUJOCO_GL=.*/export MUJOCO_GL=$backend/" "$HOME/.bashrc"
      echo "rendering works with MUJOCO_GL=$backend"
      render_ok=1; break
    fi
    if [[ "$backend" == egl && ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
      echo 'installing NVIDIA EGL ICD so glvnd finds the WSL driver'
      sudo mkdir -p /usr/share/glvnd/egl_vendor.d
      echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null
      if MUJOCO_GL=egl python "$REPO/scripts/check_render.py"; then
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

  step 'phase2: JAX with CUDA 12'
  uv pip install -U "jax[cuda12]"
  python - <<'PY'
import jax
devs = jax.devices()
print("jax", jax.__version__, "devices:", devs)
if not any(d.platform == "gpu" for d in devs):
    raise SystemExit("FATAL: JAX sees no GPU. Check nvidia-smi inside WSL and the driver passthrough.")
PY

  step 'phase2: MuJoCo Playground (editable clone -- we need learning/train_jax_ppo.py)'
  if [[ ! -d "$PLAYGROUND/.git" ]]; then
    git clone --depth 1 https://github.com/google-deepmind/mujoco_playground "$PLAYGROUND"
  fi
  uv pip install -e "$PLAYGROUND"
  uv pip install mujoco-mjx

  step 'phase2: verify the G1 joystick env loads'
  python "$REPO/scripts/check_phase2.py"
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
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  step 'phase6: DimOS'
  if uv pip install 'dimos[base,unitree]'; then
    python -c "import dimos; print('dimos ok')" || warn 'dimos imported badly'
  else
    warn 'dimos install failed -- Phase 6 only, does not affect Phase 2. Revisit closer to Sept.'
  fi
fi

step 'done'
cat <<EOF
Environment ready. From $REPO with the venv active:

  Phase 1 (should still pass here):
    python scripts/inspect_model.py
    python terrain/drop_test.py
    python sim/pose_and_render.py

  Phase 2:
    python scripts/check_phase2.py          # jax GPU + G1 env smoke test
    python scripts/train_g1.py --smoke      # ~2 min, proves the training loop runs
    python scripts/train_g1.py              # the real baseline run
EOF
