param(
    [string]$JetsonHost = "192.168.137.100",
    [string]$JetsonUser = "jetson",
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\asv_day12_ed25519"
)

$ErrorActionPreference = "Stop"
$identityDirectory = Split-Path -Parent $IdentityFile
New-Item -ItemType Directory -Force -Path $identityDirectory | Out-Null

if (-not (Test-Path $IdentityFile)) {
    Write-Host "Create the dedicated key. Press Enter twice for an empty passphrase."
    & ssh-keygen.exe -t ed25519 -f $IdentityFile
    if ($LASTEXITCODE -ne 0) {
        throw "ssh-keygen failed with exit code $LASTEXITCODE"
    }
}

$publicKey = (Get-Content "$IdentityFile.pub" -Raw).Trim()
if (-not $publicKey) {
    throw "Public key is empty: $IdentityFile.pub"
}

Write-Host "Enter the Jetson password once to install the dedicated key."
$publicKey | & ssh.exe "${JetsonUser}@${JetsonHost}" `
    "umask 077; mkdir -p ~/.ssh; key=`$(cat); grep -qxF `"`$key`" ~/.ssh/authorized_keys 2>/dev/null || printf '%s\n' `"`$key`" >> ~/.ssh/authorized_keys"
if ($LASTEXITCODE -ne 0) {
    throw "SSH public-key installation failed with exit code $LASTEXITCODE"
}

& ssh.exe -i $IdentityFile -o BatchMode=yes `
    "${JetsonUser}@${JetsonHost}" "echo SCENE_SSH_KEY_PASS"
if ($LASTEXITCODE -ne 0) {
    throw "SSH key verification failed with exit code $LASTEXITCODE"
}
