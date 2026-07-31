param(
    [string]$DataRoot = "",
    [string]$PythonExe = "",
    [string]$RunId = "0FCC05104CD8B7388994E9B5477ED769",
    [string]$GitSha = "eb832f3",
    [int]$SampleCount = 20,
    [double]$CosineThreshold = 0.999
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $DataRoot) {
    $DataRoot = Join-Path (Split-Path $RepoRoot -Parent) "pc_datasets"
}
if (-not $PythonExe) {
    $PythonExe = Join-Path $DataRoot ".venv\Scripts\python.exe"
}

$BundleName = "day12_L2_S0_R1_$RunId"
$BundleRoot = Join-Path $DataRoot "extracted\$BundleName"
$Episode = Join-Path $BundleRoot "artifacts\day8_episode\$RunId"
$Supervision = Join-Path $BundleRoot "artifacts\day10_supervised\$RunId"
$Instructions = Join-Path $RepoRoot "dataset\language\instructions.jsonl"
$LanguageModel = Join-Path $DataRoot "models\Qwen3-Embedding-0.6B"
$OutputRoot = Join-Path $DataRoot "features_pc_$GitSha"
$PcCache = Join-Path $OutputRoot $RunId
$JetsonReference = Join-Path $DataRoot "features_reference\$GitSha\$RunId"

$RequiredPaths = @(
    $PythonExe,
    $Episode,
    $Supervision,
    $Instructions,
    $LanguageModel,
    $JetsonReference
)
foreach ($Path in $RequiredPaths) {
    if (-not (Test-Path $Path)) {
        throw "Required training path does not exist: $Path"
    }
}

$env:PYTHONPATH = Join-Path $RepoRoot "src\asv_vla"
$env:USE_TF = "0"
$env:USE_FLAX = "0"
$env:USE_TORCH = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TORCH_HOME = Join-Path $DataRoot "models\torch"

function Invoke-TrainingPython {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    $ReportRoot = Join-Path $DataRoot "reports"
    New-Item -ItemType Directory -Force $ReportRoot | Out-Null
    $Stdout = Join-Path $ReportRoot "$Name.out"
    $Stderr = Join-Path $ReportRoot "$Name.err"
    Remove-Item $Stdout, $Stderr -Force -ErrorAction SilentlyContinue
    $Process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $RepoRoot `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr
    Get-Content $Stdout, $Stderr -ErrorAction SilentlyContinue
    if ($Process.ExitCode -ne 0) {
        throw "$Name failed with exit code $($Process.ExitCode)"
    }
}

Invoke-TrainingPython -Name "pc_build" -Arguments @(
    "-m", "training.feature_cache", "build",
    "--episode", $Episode,
    "--supervision", $Supervision,
    "--instructions", $Instructions,
    "--output-root", $OutputRoot,
    "--language-model-path", $LanguageModel,
    "--language-weights-sha256",
    "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    "--visual-weights-sha256", "auto",
    "--git-sha", $GitSha,
    "--device", "cuda",
    "--cuda-load-attempts", "2"
)

Invoke-TrainingPython -Name "pc_compare" -Arguments @(
    "-m", "training.feature_cache", "compare",
    $PcCache,
    $JetsonReference,
    "--sample-count", "$SampleCount",
    "--cosine-threshold", "$CosineThreshold"
)
