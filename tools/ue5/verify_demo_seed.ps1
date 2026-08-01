param(
    [string]$UeProject = "D:\Unreal Projects\VLA\VLA.uproject",
    [string]$UnrealExe = "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe",
    [int]$SceneSeed = 200101,
    [string]$LayoutId = "L1",
    [string]$MotionState = "S0",
    [string]$SlotId = "CL-001",
    [int]$RunSeconds = 60,
    [double]$SineWavelength = 6000.0,
    [double]$SineAmplitude = 600.0,
    [double]$SineSpeed = 60.0,
    [double]$SineDelay = 0.0,
    [switch]$YawFixWholeRun,
    [int]$SceneExecPort = 0
)

# Headless scene verification: launch the fixed automation binary with a
# given SceneSeed and capture the UE5 log.  UE5 exits by itself after
# RunSeconds (automation MaxRuntimeSeconds).  The caller starts the Jetson
# VLA pipeline first.

$ErrorActionPreference = "Stop"

if (-not (Test-Path $UeProject)) {
    throw "UE project not found: $UeProject"
}
if (-not (Test-Path $UnrealExe)) {
    throw "UnrealEditor not found: $UnrealExe"
}

function Quote-NativeArgument([string]$Value) {
    # Start-Process joins ArgumentList before invoking the native process.
    # Keep the project path explicitly quoted so UE sees it as argv[0], even
    # though the Windows path contains spaces.
    return '"' + $Value + '"'
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $env:TEMP "demo-seed${SceneSeed}-$stamp.ue.log"
$ueArguments = @(
    (Quote-NativeArgument $UeProject),
    "/Game/Main_Map",
    "-game",
    "-SceneAuto",
    "-Slot=$SlotId",
    "-Layout=$LayoutId",
    "-Motion=$MotionState",
    "-Seed=$SceneSeed",
    "-MaxRuntimeSeconds=$RunSeconds",
    "-SineWavelength=$SineWavelength",
    "-SineAmplitude=$SineAmplitude",
    "-SineSpeed=$SineSpeed",
    "-SineDelay=$SineDelay",
    $(if ($YawFixWholeRun) { "-YawFixWholeRun" } else { "" }),
    $(if ($SceneExecPort -gt 0) { "-SceneExecPort=$SceneExecPort" } else { "" }),
    "-RenderOffscreen",
    "-unattended",
    "-nosplash",
    "-stdout",
    "-FullStdOutLogOutput"
)

$ueProcess = Start-Process $UnrealExe `
    -ArgumentList $ueArguments `
    -RedirectStandardOutput $log `
    -RedirectStandardError "$log.err" `
    -PassThru

$deadline = (Get-Date).AddSeconds($RunSeconds + 90)
while (-not $ueProcess.HasExited -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 1000
}
if (-not $ueProcess.HasExited) {
    Stop-Process -Id $ueProcess.Id -Force
    Write-Host "DEMO_UE_TIMEOUT_KILLED"
}
$ueProcess.WaitForExit()

Write-Host "DEMO_UE_LOG path=$log"
Write-Host "DEMO_UE_EXIT code=$($ueProcess.ExitCode)"
