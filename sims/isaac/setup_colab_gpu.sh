#!/usr/bin/env bash
# Colab-specific plumbing that has to happen BEFORE setup_isaac_cloud.sh.
#
#     bash sims/isaac/setup_colab_gpu.sh
#     bash sims/isaac/setup_isaac_cloud.sh          # then the normal Linux installer
#
# Idempotent: safe to re-run; each step no-ops if already done.
#
# setup_isaac_cloud.sh assumes a rented Linux box that already looks like a workstation --
# python3.11 present, a real GPU driver install with its Vulkan/EGL manifests, plenty of
# room on /. A Colab runtime is none of those things, and each gap fails in a way that does
# not name itself:
#
#   * python3.12          isaacsim has no 3.12 wheel; pip reports "Could not find a version
#                         that satisfies the requirement isaacsim" as if the package were
#                         misspelled. This is the single most common Isaac-on-Colab failure.
#   * no Vulkan ICD       Colab ships the NVIDIA driver libraries but NOT the JSON manifests
#                         that tell the Vulkan/EGL loaders where they are. Kit then finds no
#                         device and either falls back to something unusable or hangs during
#                         startup. --video cannot work at all without this.
#   * small /             the pip install unpacks ~25 GB. Left on the default TMPDIR it fills
#                         the root volume and dies partway through, leaving a half-written
#                         venv that then fails the numpy check for an unrelated-looking reason.
#
# The ICD/vendor JSON contents follow the published j3soon/isaac-sim-colab recipe, which is
# the only Colab install of Isaac Sim anyone has documented working end to end. Note that it
# targets Isaac Sim 4.5; we pin 5.1, so treat a first run here as unproven -- notes/setup.md
# and the notebook's preflight cell carry the abort criterion and the fallback.

set -euo pipefail

PY_VERSION=3.11
DRIVE_CACHE="${DRIVE_CACHE:-}"        # e.g. /content/drive/MyDrive/GeologicDome/cache

step() { printf '\n== %s ==\n' "$1"; }

step 'GPU and driver'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Advisory only -- the notebook's preflight cell is where the hard abort lives, because a
# bash exit here is easy to scroll past in a notebook. Isaac Sim 5.1's stated Linux minimum
# is 580.65.06, and NVIDIA lists GPUs without RT cores (A100, H100) as unsupported.
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [ "$(printf '%s\n580.65.06\n' "$DRIVER" | sort -V | head -1)" != "580.65.06" ]; then
    echo "  WARNING: driver $DRIVER is below Isaac Sim 5.1's stated minimum 580.65.06."
    echo "           Colab's driver cannot be upgraded. If Kit fails to start, this is why;"
    echo "           the fallback is a rented L40S/A10 box, where setup_isaac_cloud.sh runs"
    echo "           unchanged."
fi
case "$GPU" in
    *A100*|*H100*)
        echo "  WARNING: $GPU has no RT cores, which NVIDIA lists as unsupported for Isaac"
        echo "           Sim 5.1. Headless physics-only training may still work, but --video"
        echo "           brings up the offscreen RTX renderer and is the part that needs them."
        echo "           Prefer an L4 (Runtime -> Change runtime type)." ;;
esac

step "python$PY_VERSION (Isaac Sim 5.1 does not support 3.12)"
if command -v "python$PY_VERSION" >/dev/null; then
    echo "  already present: $(python$PY_VERSION --version)"
else
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -qq
    sudo apt-get install -y -qq software-properties-common >/dev/null
    sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
    sudo apt-get update -qq
    sudo apt-get install -y -qq "python$PY_VERSION" "python$PY_VERSION-venv" \
        "python$PY_VERSION-dev" >/dev/null
    echo "  installed: $(python$PY_VERSION --version)"
fi

step 'graphics libraries Kit loads at startup'
sudo apt-get install -y -qq vulkan-tools libvulkan1 libglu1-mesa libegl1 libgl1 libglx0 \
    libxt6 libgomp1 libxrandr2 libxinerama1 libxcursor1 libxi6 libsm6 libice6 >/dev/null

step 'Vulkan ICD + EGL vendor manifests (Colab ships the drivers, not these files)'
sudo mkdir -p /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d /usr/share/glvnd/egl_vendor.d

sudo tee /etc/vulkan/icd.d/nvidia_icd.json >/dev/null <<'JSON'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.194"
    }
}
JSON

sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null <<'JSON'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
JSON

# Kit probes for the Optimus layer on hybrid-graphics systems. Absent, startup logs a stream
# of loader errors that look like a driver fault; present, it is a no-op on a datacentre GPU.
sudo tee /etc/vulkan/implicit_layer.d/nvidia_layers.json >/dev/null <<'JSON'
{
    "file_format_version": "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.194",
        "implementation_version": "1",
        "description": "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": { "__NV_PRIME_RENDER_OFFLOAD": "1" },
        "disable_environment": { "DISABLE_LAYER_NV_OPTIMUS_1": "" }
    }
}
JSON

step 'gate: does Vulkan now see the GPU?'
# The real check. Writing the JSON proves nothing -- the loader has to resolve the library
# it points at, and on a runtime whose driver userspace lives somewhere unexpected it will
# not. Everything downstream (Kit startup, and --video in particular) depends on this line.
if vulkaninfo --summary 2>/dev/null | grep -qi "nvidia"; then
    vulkaninfo --summary 2>/dev/null | grep -i "deviceName\|driverVersion" | head -4
    echo "  OK — Vulkan resolves an NVIDIA device"
else
    echo "  FAILED — Vulkan does not see an NVIDIA device."
    echo "  Kit will not start a renderer, so --video is out and headless training is at risk."
    echo "  Things to try, in order:"
    echo "    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json"
    echo "    export __GLX_VENDOR_LIBRARY_NAME=nvidia"
    echo "    ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0   # is it even installed?"
    exit 1
fi

step 'scratch space on the big volume'
# pip unpacks ~25 GB. The default TMPDIR is on the small root volume on some runtimes.
mkdir -p /content/tmp
export TMPDIR=/content/tmp
grep -q 'TMPDIR=/content/tmp' "$HOME/.bashrc" 2>/dev/null \
    || echo 'export TMPDIR=/content/tmp' >> "$HOME/.bashrc"
df -h /content / | sed 's/^/  /'

step 'Omniverse EULA for non-interactive shells'
# First Kit launch otherwise prompts, which on a notebook reads as an unexplained hang.
export OMNI_KIT_ACCEPT_EULA=YES
grep -q OMNI_KIT_ACCEPT_EULA "$HOME/.bashrc" 2>/dev/null \
    || echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> "$HOME/.bashrc"

if [ -n "$DRIVE_CACHE" ]; then
    step "restore Kit shader caches from $DRIVE_CACHE"
    # Worth roughly half an hour per session. Kit compiles its shader and MDL caches on
    # first launch and a fresh Colab VM has neither, so every session pays the cold start
    # again unless the caches are carried across.
    mkdir -p "$DRIVE_CACHE"
    for pair in "ov:$HOME/.cache/ov" "ComputeCache:$HOME/.nv/ComputeCache"; do
        name="${pair%%:*}"; dest="${pair#*:}"
        if [ -d "$DRIVE_CACHE/$name" ]; then
            mkdir -p "$(dirname "$dest")"
            cp -r "$DRIVE_CACHE/$name" "$dest" 2>/dev/null || true
            echo "  restored $name"
        else
            echo "  no cached $name yet (saved after the first run)"
        fi
    done
fi

step 'done'
cat <<'EOF'

Next:

  bash sims/isaac/setup_isaac_cloud.sh      # with VENV_DIR / LAB_DIR set for /content

To save the shader caches back to Drive after a successful run, so the next session skips
the cold start:

  DRIVE_CACHE=/content/drive/MyDrive/GeologicDome/cache
  cp -r ~/.cache/ov "$DRIVE_CACHE/ov"
  cp -r ~/.nv/ComputeCache "$DRIVE_CACHE/ComputeCache"
EOF
