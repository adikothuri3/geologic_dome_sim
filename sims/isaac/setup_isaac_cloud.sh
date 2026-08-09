#!/usr/bin/env bash
# Isaac Sim 5.1.0 + Isaac Lab 2.3.2 on a rented Linux GPU box (Lambda / Brev / AWS).
# Idempotent: safe to re-run; each step no-ops if already done.
#
#     bash sims/isaac/setup_isaac_cloud.sh
#
# The Linux counterpart of setup_isaac.ps1. Same pins, same reasons (sims/isaac/README.md).
# What is deliberately NOT carried over from the Windows installer:
#   * the Norton CA bundle — that MITM is specific to the dev box
#   * the schannel git workaround — same reason
#   * `py -3.11` — Linux uses python3.11 directly
# What IS carried over, because each cost real debugging time (notes/setup.md):
#   * numpy==1.26.0    numba 0.59.1 (via isaacsim-core) requires numpy<1.27, and a 2.x
#                      wheel unpacked over a 1.x install breaks trimesh -> isaaclab_assets
#   * tensordict==0.8.3 the current wheel is built against a newer torch ABI and
#                      access-violates on import with torch 2.7.0
#   * setuptools<81   flatdict 4.0.1 has no wheel on PyPI, so pip builds it from sdist,
#                      and its setup.py does `import pkg_resources`. setuptools 81 dropped
#                      pkg_resources. --no-build-isolation alone is NOT enough -- it only
#                      moves the build into the venv, so the venv's setuptools must still
#                      carry pkg_resources. Measured 2026-08-09: setuptools 84.0.0 ->
#                      "ModuleNotFoundError: No module named 'pkg_resources'" ->
#                      metadata-generation-failed; setuptools 80.10.2 -> installs clean.
#
# Do NOT bump to Isaac Lab 3.0 beta: its Newton/kit-less branch is a different API surface.

set -euo pipefail

ISAAC_SIM_VERSION=5.1.0
LAB_REF=v2.3.2
VENV_DIR="${VENV_DIR:-$HOME/venvs/isaac}"
LAB_DIR="${LAB_DIR:-$HOME/src/IsaacLab}"

step() { printf '\n== %s ==\n' "$1"; }

step 'preflight: python3.11 and NVIDIA driver'
command -v python3.11 >/dev/null || {
    echo "python3.11 required (Isaac Sim 5.1 does not support 3.12)."
    echo "Ubuntu: sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get install -y python3.11 python3.11-venv"
    exit 1
}
python3.11 --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Kit needs these even headless — it still initialises GL/Vulkan loaders at startup.
# Non-fatal on purpose: `apt-get update` returns non-zero for a single unreachable or
# expired third-party repo, which is routine on a hosted runtime and says nothing about
# whether these libraries are installable. Killing a 25 GB install over a repo warning is
# the wrong trade, and on Colab setup_colab_gpu.sh has already installed a superset.
step 'system libraries Kit loads at startup'
if command -v apt-get >/dev/null; then
    sudo apt-get update -qq || echo '  (apt-get update returned non-zero; continuing)'
    sudo apt-get install -y -qq libglu1-mesa libxt6 libgomp1 libxrandr2 libxinerama1 \
        libxcursor1 libxi6 libsm6 libice6 >/dev/null 2>&1 \
        || echo '  (some libraries could not be installed; continuing)'
fi

# First Kit launch asks for the Omniverse EULA interactively, which hangs a headless box.
step 'accept the Omniverse EULA for non-interactive shells'
export OMNI_KIT_ACCEPT_EULA=YES
touch "$HOME/.bashrc"
grep -q OMNI_KIT_ACCEPT_EULA "$HOME/.bashrc" || echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> "$HOME/.bashrc"

step "venv at $VENV_DIR"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    python3.11 -m venv "$VENV_DIR" || {
        echo "  FAILED: could not create a venv at $VENV_DIR."
        echo "  Usually python3.11-venv is missing:  sudo apt-get install -y python3.11-venv"
        echo "  On Colab, setup_colab_gpu.sh installs and proves this before you get here."
        exit 1
    }
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install --quiet --upgrade pip wheel
# NOT --upgrade setuptools: see the flatdict note in the header.
"$PY" -m pip install --quiet "setuptools<81"

step 'disk space for the ~25 GB install'
avail_gb=$(df -BG --output=avail "$(dirname "$VENV_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "  $(dirname "$VENV_DIR"): ${avail_gb:-?} GB available (need ~30)"
if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -lt 30 ]; then
    echo "  FAILED: not enough free disk. isaacsim unpacks to ~25 GB and pip needs scratch"
    echo "  space on top. Free space or use a runtime with a bigger disk."
    exit 1
fi

step "isaacsim==$ISAAC_SIM_VERSION (~10 GB download - this is the slow step)"
if "$PY" -m pip show isaacsim >/dev/null 2>&1; then
    echo '  already installed'
else
    "$PY" -m pip install "isaacsim[all,extscache]==$ISAAC_SIM_VERSION" \
        --extra-index-url https://pypi.nvidia.com
fi

step "Isaac Lab $LAB_REF at $LAB_DIR"
mkdir -p "$(dirname "$LAB_DIR")"
[ -d "$LAB_DIR/.git" ] || git clone https://github.com/isaac-sim/IsaacLab "$LAB_DIR"
git -C "$LAB_DIR" fetch --tags --quiet
git -C "$LAB_DIR" checkout "$LAB_REF" --quiet
echo "  checked out $LAB_REF"

step 'Isaac Lab packages -> venv (direct pip; isaaclab.sh only understands conda/kit-python)'
# Re-pin here, not just at venv creation: installing isaacsim above can drag a newer
# setuptools back in, and this is the one step that cannot tolerate it.
"$PY" -m pip install --quiet "setuptools<81"
"$PY" -c "import pkg_resources" 2>/dev/null || {
    echo "  FAILED: setuptools in this venv has no pkg_resources, which flatdict's sdist"
    echo "  needs at build time. Try:  $PY -m pip install 'setuptools<81'"
    exit 1
}
"$PY" -m pip install --quiet --no-build-isolation flatdict==4.0.1
for pkg in source/isaaclab source/isaaclab_assets "source/isaaclab_rl[rsl-rl]" source/isaaclab_tasks; do
    echo "  pip install -e $pkg"
    "$PY" -m pip install --quiet --editable "$LAB_DIR/$pkg"
done

step 'version pins that are load-bearing (see header)'
"$PY" -m pip install --quiet "tensordict==0.8.3" "numpy==1.26.0"
# The real gate for the numpy pin: `import numpy` alone passes on a half-overwritten tree.
"$PY" -c "import numpy, numpy.ma.mrecords, trimesh; print('  numpy', numpy.__version__, '+ trimesh import OK')"

"$PY" -c "from importlib.metadata import version; [print(f'  {p:>16}', version(p)) for p in ('isaacsim','isaaclab','isaaclab-tasks','isaaclab-assets','isaaclab-rl','rsl-rl-lib')]"

step 'done'
cat <<EOF

Next, in order (sims/isaac/README.md § The cloud run has the full walkthrough):

  export OMNI_KIT_ACCEPT_EULA=YES
  PY=$PY

  # 1. Domain randomization actually reaches PhysX. Seconds, and it is the difference
  #    between paying for a randomized policy and one trained on a single fixed robot.
  \$PY sims/isaac/scripts/check_isaac.py --gate c --num_envs 32

  # 2. The real run.
  \$PY sims/isaac/scripts/train_g1_flat.py --num_envs 4096 --max_iterations 1500

  # 3. Score it, render it, plot it.
  \$PY sims/isaac/scripts/play_g1_flat.py runs/isaac/<run_id> --video
  \$PY sims/isaac/scripts/plot_play.py    runs/isaac/<run_id>
EOF
