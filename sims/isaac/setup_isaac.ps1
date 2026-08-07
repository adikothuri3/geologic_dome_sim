# Isaac Sim 5.1.0 + Isaac Lab 2.3.x installer for this box (Windows 11, RTX 4060 Ti 8 GB).
# Idempotent: safe to re-run; each step no-ops if already done.
#
#     powershell -ExecutionPolicy Bypass -File sims\isaac\setup_isaac.ps1
#
# Versions are pinned on purpose -- see sims/isaac/README.md. Do NOT bump to Isaac Lab 3.0
# beta (Ubuntu-only as of mid-2026).
#
# This box is BELOW Isaac's official minimum (8 GB VRAM vs 16, 16 GB RAM vs 32).
# Local = headless prototyping only; real training goes to a cloud GPU.

$ErrorActionPreference = 'Stop'

# Norton Antivirus MITMs TLS on this box: Windows tools trust its root, Python's OpenSSL
# does not. The bundle below is pip's vendored certifi + the exported Norton root
# (see notes/setup.md). Without it every pip call dies with CERTIFICATE_VERIFY_FAILED.
$CaBundle = Join-Path $env:USERPROFILE 'venvs\ca-bundle-norton.pem'
if (Test-Path $CaBundle) { $env:PIP_CERT = $CaBundle; $env:SSL_CERT_FILE = $CaBundle }

$IsaacSimVersion = '5.1.0'
$VenvDir  = Join-Path $env:USERPROFILE 'venvs\isaac'
$SrcDir   = Join-Path $env:USERPROFILE 'src'
$LabDir   = Join-Path $SrcDir 'IsaacLab'
$LabRepo  = 'https://github.com/isaac-sim/IsaacLab'
# Isaac Lab 2.3.x line: main is in maintenance for 2.3; pin the release tag for reproducibility.
$LabRef   = 'v2.3.2'

function Step($msg) { Write-Host "`n== $msg ==" -ForegroundColor Cyan }

# ---------------------------------------------------------------- preflight --
Step 'preflight: python 3.11 and NVIDIA driver'
try { $py = & py -3.11 --version } catch { $py = $null }
if (-not $py) {
    Write-Host 'Python 3.11 x64 is required (Isaac Sim 5.1 does not support 3.12).'
    Write-Host 'Install from https://www.python.org/downloads/ then re-run.'
    exit 1
}
Write-Host "  $py"
$smi = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if (-not $smi) { Write-Host 'nvidia-smi not found -- NVIDIA driver missing?'; exit 1 }
Write-Host "  $smi"

# --------------------------------------------------------------------- venv --
Step "venv at $VenvDir"
if (Test-Path (Join-Path $VenvDir 'Scripts\python.exe')) {
    Write-Host '  reusing existing venv'
} else {
    & py -3.11 -m venv $VenvDir
}
$VPy = Join-Path $VenvDir 'Scripts\python.exe'
& $VPy -m pip install --upgrade pip | Out-Null

# ---------------------------------------------------------------- isaac sim --
Step "isaacsim==$IsaacSimVersion (pip, ~10 GB download -- this takes a while)"
# PS 5.1: redirecting a native command's stderr under ErrorActionPreference=Stop turns
# pip's harmless "not found" stderr line into a terminating error -- probe via exit code.
$prevEap = $ErrorActionPreference; $ErrorActionPreference = 'SilentlyContinue'
& $VPy -m pip show isaacsim *>$null
$ErrorActionPreference = $prevEap
if ($LASTEXITCODE -eq 0) {
    Write-Host '  already installed'
} else {
    & $VPy -m pip install "isaacsim[all,extscache]==$IsaacSimVersion" --extra-index-url https://pypi.nvidia.com
}

# ---------------------------------------------------------------- isaac lab --
Step "Isaac Lab ($LabRef) at $LabDir"
if (-not (Test-Path $SrcDir)) { New-Item -ItemType Directory $SrcDir | Out-Null }
# git's global config pins OpenSSL, which can't verify through Norton's TLS MITM --
# use Windows Schannel for these calls (same workaround notes/setup.md documents).
if (-not (Test-Path (Join-Path $LabDir '.git'))) {
    git -c http.sslBackend=schannel clone $LabRepo $LabDir
}
git -C $LabDir -c http.sslBackend=schannel fetch --tags --quiet
git -C $LabDir checkout $LabRef --quiet
Write-Host "  checked out $LabRef"

Step 'isaaclab.bat --install (installs isaaclab pkgs + RSL-RL into the venv)'
# isaaclab.bat installs into the ACTIVE python environment -- activate the venv first,
# or everything lands in system Python.
& (Join-Path $VenvDir 'Scripts\Activate.ps1')
Push-Location $LabDir
try {
    & .\isaaclab.bat --install
} finally {
    Pop-Location
}

# --------------------------------------------------------------------- done --
Step 'done -- smoke ladder (run these YOURSELF, in order; each is a gate)'
Write-Host @"

  # (a) Isaac Sim opens headless and closes clean. FIRST RUN IS SLOW (shader cache).
  $VPy -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); app.close()"

  # (b) Built-in G1 velocity task loads and steps (from $LabDir):
  $VPy scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 8

  # (c) 10-iteration training smoke, logs under runs\isaac\:
  $VPy scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 64 --max_iterations 10

Remember: 8 GB VRAM / 16 GB RAM is below Isaac's minimum. Headless always; num_envs small;
real training on a cloud GPU. Log every run as a notes/experiments.md row.
"@
