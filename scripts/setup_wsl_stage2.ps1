<#
Stage 2 of the WSL setup -- run AFTER the reboot that follows stage 1.

Stage 1 (already done, needed elevation): enabled Microsoft-Windows-Subsystem-Linux and
VirtualMachinePlatform, installed the WSL runtime (2.7.11).

This stage installs the Ubuntu 24.04 distro, creates a normal user non-interactively, and
hands off to scripts/setup_wsl.sh inside WSL for the Linux-side toolchain.

    powershell -ExecutionPolicy Bypass -File scripts\setup_wsl_stage2.ps1

Does NOT need elevation: registering a distro is a per-user operation once the WSL runtime
is installed.
#>

param(
    [string]$Distro   = 'Ubuntu-24.04',
    [string]$UnixUser = 'aditya',
    [switch]$SkipLinuxSetup
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

function Step($m) { Write-Host "`n== $m ==" -ForegroundColor Cyan }

Step 'preflight'
$virt = (Get-CimInstance Win32_Processor).VirtualizationFirmwareEnabled
if (-not $virt) {
    throw "AMD SVM is still disabled in firmware. Reboot into BIOS (shutdown /r /fw /t 0), Tweaker -> Advanced CPU Settings -> SVM Mode -> Enabled."
}
Write-Host "SVM enabled           : $virt"
Write-Host "HypervisorPresent     : $((Get-CimInstance Win32_ComputerSystem).HypervisorPresent)"

# If the features were enabled but the machine was never rebooted, wsl.exe still errors out.
$wslOk = $false
try { wsl.exe --status *> $null; $wslOk = ($LASTEXITCODE -eq 0) } catch { $wslOk = $false }
if (-not $wslOk) {
    throw "wsl.exe is not functional yet. Stage 1 enabled the Windows features but they need a reboot. Reboot and re-run this script."
}
wsl.exe --version

Step "installing $Distro"
$installed = (wsl.exe --list --quiet) -replace "`0", '' | Where-Object { $_ -match [regex]::Escape($Distro) }
if ($installed) {
    Write-Host "$Distro already registered"
} else {
    # --no-launch: the default first run opens an interactive console asking for a UNIX
    # username and password, which would block forever in an unattended script.
    wsl.exe --install -d $Distro --no-launch
    if ($LASTEXITCODE -ne 0) { throw "wsl --install -d $Distro failed with $LASTEXITCODE" }
}

Step 'registering root filesystem'
# First invocation as root completes OOBE without prompting for a user.
wsl.exe -d $Distro --user root -- /bin/true
if ($LASTEXITCODE -ne 0) { throw "could not start $Distro as root" }

Step "creating unix user '$UnixUser'"
$mkuser = @"
set -e
if ! id -u $UnixUser >/dev/null 2>&1; then
  adduser --disabled-password --gecos '' $UnixUser
  usermod -aG sudo $UnixUser
  echo '$UnixUser ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-$UnixUser
  chmod 0440 /etc/sudoers.d/90-$UnixUser
fi
printf '[user]\ndefault=$UnixUser\n\n[boot]\nsystemd=true\n' > /etc/wsl.conf
echo "user ok: \$(id $UnixUser)"
"@
$mkuser = $mkuser -replace "`r`n", "`n"
wsl.exe -d $Distro --user root -- bash -lc $mkuser
if ($LASTEXITCODE -ne 0) { throw 'user creation failed' }

Step 'applying wsl.conf (terminate + restart distro)'
wsl.exe --terminate $Distro | Out-Null
wsl.exe -d $Distro -- bash -lc 'echo "default user is now: $(whoami)"'

Step 'GPU passthrough check'
wsl.exe -d $Distro -- bash -lc 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "WARN: nvidia-smi unavailable inside WSL"'

if ($SkipLinuxSetup) {
    Write-Host "`nSkipping Linux setup (-SkipLinuxSetup). Run it yourself with:" -ForegroundColor Yellow
    Write-Host "    wsl -d $Distro -- bash -lc 'cd /mnt/c/Users/Aditya/VSCode/GeologicDome && bash scripts/setup_wsl.sh --all'"
    exit 0
}

Step 'linux toolchain (scripts/setup_wsl.sh --all)'
$repoWsl = (wsl.exe -d $Distro -- wslpath -a ($repo -replace '\\', '/')) -replace "`0", ''
$repoWsl = $repoWsl.Trim()
Write-Host "repo inside WSL: $repoWsl"
wsl.exe -d $Distro -- bash -lc "cd '$repoWsl' && bash scripts/setup_wsl.sh --all"
exit $LASTEXITCODE
