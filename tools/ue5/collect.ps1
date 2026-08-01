param(
    [ValidateRange(0, 30)]
    [int]$Count = 1,
    [string]$JetsonHost = "192.168.137.100",
    [string]$JetsonUser = "jetson",
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\asv_day12_ed25519",
    [string]$RemoteRepo = "/home/jetson/jetson_asv_ws",
    [string]$RemotePlan = "training/config/collection_plan_30_v1.json",
    [string]$UeProject = "D:\Unreal Projects\VLA\VLA.uproject",
    [string]$UnrealExe = "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe",
    [string]$LocalOutput = "",
    [string]$ExecutionAddress = "192.168.137.1",
    [int]$ExecutionPort = 8081,
    [double]$MaxSpeedMps = 0.8,
    [int]$ReadyTimeoutSeconds = 90,
    [int]$RunTimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$sshExe = Join-Path $env:WINDIR "System32\OpenSSH\ssh.exe"
$scpExe = Join-Path $env:WINDIR "System32\OpenSSH\scp.exe"

function Quote-NativeArgument([string]$Value) {
    # UE requires the .uproject path as the first native argument.  Preserve
    # the quotes through Start-Process so spaces in D:\Unreal Projects\... do
    # not split the project path into multiple argv entries.
    return '"' + $Value + '"'
}

if (-not $LocalOutput) {
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
    $LocalOutput = Join-Path (Split-Path -Parent $repoRoot) "pc_datasets"
}
if (-not (Test-Path $IdentityFile)) {
    throw "SSH key not found: $IdentityFile. Run setup_ssh.ps1 once."
}
if (-not (Test-Path $UeProject)) {
    throw "UE project not found: $UeProject"
}
if (-not (Test-Path $UnrealExe)) {
    throw "UnrealEditor not found: $UnrealExe"
}
if (-not (Test-Path $sshExe) -or -not (Test-Path $scpExe)) {
    throw "Windows OpenSSH client is not installed."
}
if (Get-Process UnrealEditor -ErrorAction SilentlyContinue) {
    throw "Close the interactive UnrealEditor before unattended collection."
}
New-Item -ItemType Directory -Force -Path $LocalOutput | Out-Null

$sshTarget = "${JetsonUser}@${JetsonHost}"
$completedThisInvocation = 0

while ($Count -eq 0 -or $completedThisInvocation -lt $Count) {
    $nextCommand = "cd '$RemoteRepo' && python3 -m training.collection next --data-root . --plan '$RemotePlan' --json"
    $queryStamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
    $queryStdout = Join-Path $env:TEMP "collect-query-$queryStamp.stdout.log"
    $queryStderr = Join-Path $env:TEMP "collect-query-$queryStamp.stderr.log"
    $queryProcess = Start-Process $sshExe `
        -ArgumentList @(
            "-i", (Quote-NativeArgument $IdentityFile),
            "-o", "BatchMode=yes",
            $sshTarget,
            (Quote-NativeArgument $nextCommand)
        ) `
        -RedirectStandardOutput $queryStdout `
        -RedirectStandardError $queryStderr `
        -Wait `
        -PassThru
    $nextOutput = Get-Content $queryStdout
    if ($queryProcess.ExitCode -ne 0) {
        $queryErrors = Get-Content $queryStderr -Raw -ErrorAction SilentlyContinue
        throw "Cannot query the next collection slot from Jetson.`n$queryErrors"
    }
    $nextLine = @($nextOutput | Where-Object { $_.Trim() })[-1]
    $next = $nextLine | ConvertFrom-Json
    if ($next.complete) {
        Write-Host "SCENE_BATCH_COMPLETE no pending collection slots"
        break
    }

    $slot = [string]$next.slot_id
    $layout = [string]$next.layout_id
    $motion = [string]$next.motion_state
    $seed = [int]$next.scene_seed
    $rolloutAction = [string]$next.rollout_action
    if (-not $rolloutAction) {
        $rolloutAction = "follow"
    }
    Write-Host "SCENE_BATCH_SLOT slot=$slot layout=$layout motion=$motion scene_seed=$seed rollout_action=$rolloutAction"

    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $remoteStdout = Join-Path $env:TEMP "collect-$slot-$stamp.stdout.log"
    $remoteStderr = Join-Path $env:TEMP "collect-$slot-$stamp.stderr.log"
    $ueStdout = Join-Path $env:TEMP "collect-$slot-$stamp.ue.stdout.log"
    $ueStderr = Join-Path $env:TEMP "collect-$slot-$stamp.ue.stderr.log"
    $remoteCommand = "cd '$RemoteRepo' && EXECUTION_ADDRESS='$ExecutionAddress' EXECUTION_PORT='$ExecutionPort' MAX_SPEED_MPS='$MaxSpeedMps' bash scripts/remote_collect.sh '$slot' '$layout' '$motion' '$seed' '$RemotePlan' '$rolloutAction'"

    $remoteProcess = Start-Process $sshExe `
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
        if ($output -match "SCENE_REMOTE_READY") {
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
        "-SceneAuto",
        "-Slot=$slot",
        "-Layout=$layout",
        "-Motion=$motion",
        "-Seed=$seed",
        "-MaxRuntimeSeconds=$RunTimeoutSeconds",
        "-SceneExecPort=$ExecutionPort",
        "-YawFixWholeRun",
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
            throw "Collection slot timed out after $RunTimeoutSeconds seconds."
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
    if ($output -notmatch "SCENE_REMOTE_COMPLETE") {
        throw "Jetson collection did not emit its success marker.`n$errors"
    }

    $packageMatch = [regex]::Match(
        $output,
        "SCENE_PACKAGE path=(\S+) sha256=([0-9a-f]{64})"
    )
    if (-not $packageMatch.Success) {
        throw "Jetson completed without a SCENE_PACKAGE marker."
    }
    $remotePackage = $packageMatch.Groups[1].Value
    $expectedSha = $packageMatch.Groups[2].Value

    $scpStdout = Join-Path $env:TEMP "collect-$slot-$stamp.scp.stdout.log"
    $scpStderr = Join-Path $env:TEMP "collect-$slot-$stamp.scp.stderr.log"
    $scpProcess = Start-Process $scpExe `
        -ArgumentList @(
            "-i", (Quote-NativeArgument $IdentityFile),
            "-o", "BatchMode=yes",
            (Quote-NativeArgument "${sshTarget}:$remotePackage"),
            (Quote-NativeArgument $LocalOutput)
        ) `
        -RedirectStandardOutput $scpStdout `
        -RedirectStandardError $scpStderr `
        -Wait `
        -PassThru
    if ($scpProcess.ExitCode -ne 0) {
        $scpErrors = Get-Content $scpStderr -Raw -ErrorAction SilentlyContinue
        Write-Error $scpErrors
        throw "SCP failed for $remotePackage"
    }
    $localPackage = Join-Path $LocalOutput (Split-Path -Leaf $remotePackage)
    $actualSha = (Get-FileHash -Algorithm SHA256 $localPackage).Hash.ToLowerInvariant()
    if ($actualSha -ne $expectedSha) {
        throw "SHA-256 mismatch for $localPackage"
    }

    Write-Host "SCENE_PC_PACKAGE_PASS path=$localPackage sha256=$actualSha"
    $completedThisInvocation += 1
}

Write-Host "SCENE_BATCH_EXIT completed=$completedThisInvocation"
