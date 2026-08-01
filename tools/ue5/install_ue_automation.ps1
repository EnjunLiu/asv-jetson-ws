param(
    [string]$UeProject = "D:\Unreal Projects\VLA\VLA.uproject",
    [string]$EngineRoot = "D:\Softwares\Unreal Engine\UE_5.6",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $UeProject
$projectSource = Join-Path $projectRoot "Source\EDGE"
$automationSource = Join-Path $PSScriptRoot "Source\EDGE"
$dotnet = Join-Path $EngineRoot "Engine\Binaries\ThirdParty\DotNet\8.0.300\win-x64\dotnet.exe"
$unrealBuildTool = Join-Path $EngineRoot "Engine\Binaries\DotNET\UnrealBuildTool\UnrealBuildTool.dll"
$engineSource = Join-Path $EngineRoot "Engine\Source"

if (-not (Test-Path $UeProject)) {
    throw "UE project not found: $UeProject"
}
if (-not (Test-Path $automationSource)) {
    throw "Automation source not found: $automationSource"
}

New-Item -ItemType Directory -Force -Path $projectSource | Out-Null
Copy-Item -Force `
    (Join-Path $automationSource "SceneAutomationSubsystem.h") `
    (Join-Path $projectSource "SceneAutomationSubsystem.h")
Copy-Item -Force `
    (Join-Path $automationSource "SceneAutomationSubsystem.cpp") `
    (Join-Path $projectSource "SceneAutomationSubsystem.cpp")
Copy-Item -Force `
    (Join-Path $automationSource "EDGE.Build.cs") `
    (Join-Path $projectSource "EDGE.Build.cs")

Write-Host "SCENE_UE_INSTALL_PASS source=$projectSource"

if ($SkipBuild) {
    exit 0
}
if (-not (Test-Path $dotnet)) {
    throw "UE bundled dotnet not found: $dotnet"
}
if (-not (Test-Path $unrealBuildTool)) {
    throw "UnrealBuildTool not found: $unrealBuildTool"
}
if (Get-Process UnrealEditor -ErrorAction SilentlyContinue) {
    throw "UnrealEditor is running. Save work, close it once, then rerun this installer."
}

$buildProcess = Start-Process `
    -FilePath $dotnet `
    -ArgumentList @(
        ('"' + $unrealBuildTool + '"'),
        "EDGEEditor",
        "Win64",
        "Development",
        ('"' + $UeProject + '"'),
        "-WaitMutex",
        "-NoHotReloadFromIDE"
    ) `
    -WorkingDirectory $engineSource `
    -Wait `
    -PassThru `
    -NoNewWindow
if ($buildProcess.ExitCode -ne 0) {
    throw "UE build failed with exit code $($buildProcess.ExitCode)"
}

Write-Host "SCENE_UE_BUILD_PASS target=EDGEEditor"
