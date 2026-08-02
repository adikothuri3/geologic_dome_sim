#!/usr/bin/env bash
# Phase 1 environment setup -- install steps A2 through A5.
#
# Run INSIDE WSL, after `wsl --install -d Ubuntu-24.04` (step A1) and a reboot:
#
#     cd /mnt/c/Users/Aditya/VSCode/GeologicDome
#     bash scripts/setup_wsl.sh
#
# Pass --with-jax to also do step A6 (JAX-CUDA, Phase 2 prep). That step is deliberately
# NOT part of the default run: nothing in Phase 1 needs a GPU, and a failed CUDA install
# must not block the Aug 9 demo.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/venvs/dome"
MENAGERIE="$HOME/src/menagerie"
WITH_JAX=0
[[ "${1:-}" == "--with-jax" ]] && WITH_JAX=1

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

step "A2  system libraries"
sudo apt-get update
sudo apt-get install -y \
  build-essential git ffmpeg \
  libegl1 libgl1 libglx-mesa0 libosmesa6 libglfw3

step "A3  python environment"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install mujoco numpy mediapy
python -c "import mujoco; print('mujoco', mujoco.__version__)"

step "A4  menagerie (sparse: unitree_g1 only)"
if [[ ! -d "$MENAGERIE/.git" ]]; then
  mkdir -p "$(dirname "$MENAGERIE")"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie "$MENAGERIE"
fi
git -C "$MENAGERIE" sparse-checkout set unitree_g1
ls "$MENAGERIE/unitree_g1/g1.xml" "$MENAGERIE/unitree_g1/scene.xml" >/dev/null
echo "menagerie ok: $MENAGERIE/unitree_g1"

# sim/scene_g1_hfield.xml includes menagerie/unitree_g1/g1.xml relative to itself.
# Symlinks on the NTFS mount are unreliable, so fall back to a copy.
if ln -sfn "$MENAGERIE" "$REPO/sim/menagerie" 2>/dev/null && [[ -e "$REPO/sim/menagerie/unitree_g1/g1.xml" ]]; then
  echo "linked  sim/menagerie -> $MENAGERIE"
else
  echo "symlink unavailable on this filesystem; copying instead"
  rm -rf "$REPO/sim/menagerie"
  mkdir -p "$REPO/sim/menagerie"
  cp -r "$MENAGERIE/unitree_g1" "$REPO/sim/menagerie/"
  echo "copied  sim/menagerie/unitree_g1"
fi

step "A5  offscreen rendering backend"
grep -q 'MUJOCO_GL' "$HOME/.bashrc" || echo 'export MUJOCO_GL=egl' >> "$HOME/.bashrc"
grep -q "venvs/dome" "$HOME/.bashrc" || echo "source $VENV/bin/activate" >> "$HOME/.bashrc"

export MENAGERIE_DIR="$MENAGERIE"
render_ok=0
for backend in egl osmesa; do
  echo "--- trying MUJOCO_GL=$backend"
  if MUJOCO_GL="$backend" python "$REPO/scripts/check_render.py"; then
    sed -i "s/^export MUJOCO_GL=.*/export MUJOCO_GL=$backend/" "$HOME/.bashrc"
    echo "rendering works with MUJOCO_GL=$backend (persisted to ~/.bashrc)"
    render_ok=1
    break
  fi
  echo "!!! $backend failed"
  if [[ "$backend" == "egl" && ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
    echo "    hint: NVIDIA ICD file missing. To let glvnd find the WSL driver, run:"
    echo "    sudo mkdir -p /usr/share/glvnd/egl_vendor.d && \\"
    echo "      echo '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libEGL_nvidia.so.0\"}}' \\"
    echo "      | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json"
  fi
done
[[ $render_ok -eq 1 ]] || { echo "FATAL: no working GL backend. Phase 1 is blocked here."; exit 1; }

if [[ $WITH_JAX -eq 1 ]]; then
  step "A6  JAX-CUDA (Phase 2 prep, non-blocking)"
  uv pip install -U "jax[cuda12]" || echo "WARN: jax install failed -- deferred to Phase 2"
  python -c "import jax; print('jax devices:', jax.devices())" \
    || echo "WARN: no CUDA device visible to JAX -- deferred to Phase 2, Phase 1 is unaffected"
fi

step "done"
cat <<EOF
Next, from $REPO with the venv active:

    python scripts/inspect_model.py     # fluency gate: nu==29, nq==36, 'stand' present
    python terrain/drop_test.py         # terrain gate: spheres rest cleanly
    python sim/pose_and_render.py       # the demo -> phase1_stand.mp4
EOF
