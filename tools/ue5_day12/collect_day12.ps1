param(
    [ValidateRange(0, 12)]
    [int]$Count = 1,
    [string]$JetsonHost = "192.168.137.100",
    [string]$JetsonUser = "jetson",
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\asv_day12_ed25519",
    [string]$RemoteRepo = "/home/jetson/jetson_asv_ws",
    [string]$UeProject = "D:\Unreal Projects\VLA\VLA.uproject",
    [string]$UnrealExe = "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe",
    [string]$LocalOutput = "",
    [int]$ReadyTimeoutSeconds = 90,
    [int]$RunTimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"

function Quote-NativeArgument([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

if (-not $LocalOutput) {
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
    $LocalOutput = Join-Path (Split-Path -Parent $repoRoot) "pc_datasets"
}
if (-not (Test-Path $IdentityFile)) {
    throw "SSH key not found: $IdentityFile. Run setup_day12_ssh.ps1 once."
}
if (-not (Test-Path $UeProject)) {
    throw "UE project not found: $UeProject"
}
if (-not (Test-Path $UnrealExe)) {
    throw "UnrealEditor not found: $UnrealExe"
}
if (Get-Process UnrealEditor -ErrorAction SilentlyContinue) {
    throw "Close the interactive UnrealEditor before unattended collection."
}
New-Item -ItemType Directory -Force -Path $LocalOutput | Out-Null

$sshTarget = "${JetsonUser}@${JetsonHost}"
$completedThisInvocation = 0

while ($Count -eq 0 -or $completedThisInvocation -lt $Count) {
    $nextCommand = "cd '$RemoteRepo' && python3 -m training.day12_collection next --data-root . --json"
    $nextOutput = & ssh.exe -i $IdentityFile -o BatchMode=yes `
        $sshTarget $nextCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot query the next Day12 slot from Jetson."
    }
    $nextLine = @($nextOutput | Where-Object { $_.Trim() })[-1]
    $next = $nextLine | ConvertFrom-Json
    if ($next.complete) {
        Write-Host "DAY12_BATCH_COMPLETE no pending collection slots"
        break
    }

    $slot = [string]$next.slot_id
    $layout = [string]$next.layout_id
    $motion = [string]$next.motion_state
    $seed = [int]$next.scene_seed
    Write-Host "DAY12_BATCH_SLOT slot=$slot layout=$layout motion=$motion scene_seed=$seed"

    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $remoteStdout = Join-Path $env:TEMP "day12-$slot-$stamp.stdout.log"
    $remoteStderr = Join-Path $env:TEMP "day12-$slot-$stamp.stderr.log"
    $ueStdout = Join-Path $env:TEMP "day12-$slot-$stamp.ue.stdout.log"
    $ueStderr = Join-Path $env:TEMP "day12-$slot-$stamp.ue.stderr.log"
    $remoteCommand = "cd '$RemoteRepo' && bash scripts/day12_remote_collect.sh '$slot' '$layout' '$motion' '$seed'"

    $remoteProcess = Start-Process ssh.exe `
        -ArgumentList @(
            "-i", (Quote-NativeArgument $IdentityFile),
            "-o", "BatchMode=yes",
            $sshTarget,
            (Quote-NativeArgument $remoteCommand)
        ) `
        -RedirectStandardOutput $remoteStdout `
        -RedirectStandardError $remoteStderr `
        -PassThru

    $readyDeadline = (Get-Date).AddSeconds($ReadyTimeoutSeconds)
    $ready = $false
    while ((Get-Date) -lt $readyDeadline) {
        if ($remoteProcess.HasExited) {
            $output = Get-Content $remoteStdout -Raw -ErrorAction SilentlyContinue
            $errors = Get-Content $remoteStderr -Raw -ErrorAction SilentlyContinue
            throw "Jetson exited before readiness.`n$output`n$errors"
        }
        $output = Get-Content $remoteStdout -Raw -ErrorAction SilentlyContinue
        if ($output -match "DAY12_REMOTE_READY") {
            $ready = $true
            break
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) {
        Stop-Process -Id $remoteProcess.Id -Force -ErrorAction SilentlyContinue
        throw "Timed out waiting for Jetson readiness. Log: $remoteStdout"
    }

    $ueArguments = @(
        (Quote-NativeArgument $UeProject),
        "/Game/Main_Map",
        "-game",
        "-Day12Auto",
        "-Day12Slot=$slot",
        "-Day12Layout=$layout",
        "-Day12Motion=$motion",
        "-Day12Seed=$seed",
        "-Day12MaxRuntimeSeconds=$RunTimeoutSeconds",
        "-RenderOffscreen",
        "-unattended",
        "-nosplash",
        "-stdout",
        "-FullStdOutLogOutput"
    )
    $ueProcess = $null
    try {
        $ueProcess = Start-Process $UnrealExe `
            -ArgumentList $ueArguments `
            -RedirectStandardOutput $ueStdout `
            -RedirectStandardError $ueStderr `
            -PassThru

        $runDeadline = (Get-Date).AddSeconds($RunTimeoutSeconds)
        while (-not $remoteProcess.HasExited -and (Get-Date) -lt $runDeadline) {
            if ($ueProcess.HasExited) {
                $ueLog = Get-Content $ueStdout -Raw -ErrorAction SilentlyContinue
                throw "UE5 exited before Jetson completed.`n$ueLog"
            }
            Start-Sleep -Milliseconds 500
        }
        if (-not $remoteProcess.HasExited) {
            Stop-Process -Id $remoteProcess.Id -Force -ErrorAction SilentlyContinue
            throw "Day12 slot timed out after $RunTimeoutSeconds seconds."
        }
    }
    finally {
        if ($null -ne $ueProcess -and -not $ueProcess.HasExited) {
            Stop-Process -Id $ueProcess.Id -Force
            $ueProcess.WaitForExit()
        }
    }

    $remoteProcess.WaitForExit()
    $output = Get-Content $remoteStdout -Raw
    $errors = Get-Content $remoteStderr -Raw -ErrorAction SilentlyContinue
    Write-Host $output
    if ($remoteProcess.ExitCode -ne 0) {
        throw "Jetson collection failed with exit code $($remoteProcess.ExitCode).`n$errors"
    }

    $packageMatch = [regex]::Match(
        $output,
        "DAY12_PACKAGE path=(\S+) sha256=([0-9a-f]{64})"
    )
    if (-not $packageMatch.Success) {
        throw "Jetson completed without a DAY12_PACKAGE marker."
    }
    $remotePackage = $packageMatch.Groups[1].Value
    $expectedSha = $packageMatch.Groups[2].Value

    & scp.exe -i $IdentityFile -o BatchMode=yes `
        "${sshTarget}:$remotePackage" $LocalOutput
    if ($LASTEXITCODE -ne 0) {
        throw "SCP failed for $remotePackage"
    }
    $localPackage = Join-Path $LocalOutput (Split-Path -Leaf $remotePackage)
    $actualSha = (Get-FileHash -Algorithm SHA256 $localPackage).Hash.ToLowerInvariant()
    if ($actualSha -ne $expectedSha) {
        throw "SHA-256 mismatch for $localPackage"
    }

    Write-Host "DAY12_PC_PACKAGE_PASS path=$localPackage sha256=$actualSha"
    $completedThisInvocation += 1
}

Write-Host "DAY12_BATCH_EXIT completed=$completedThisInvocation"
