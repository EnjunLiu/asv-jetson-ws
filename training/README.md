# PC Training Pipeline

Day 11B PC data registry, Run-level splits, and the foundation for Day 13–16
feature caching, training and evaluation.

## Directory conventions

```text
C:\Users\LIU\Documents\asv_vla_pc\
├── repo\                         # Git checkout — only code lives here
│   ├── training\                 # THIS directory
│   │   ├── config\
│   │   │   └── dataset_v1.yaml
│   │   ├── dataset_registry.py
│   │   ├── make_group_splits.py
│   │   ├── test\
│   │   │   └── test_group_splits.py
│   │   └── README.md
│   └── ...
└── data\                         # NEVER in Git
    ├── incoming\                 # Raw Jetson tar.gz
    ├── extracted\                # Unpacked episodes + supervision
    │   └── artifacts\
    │       ├── day8_episode\<RUN_ID>\
    │       └── day10_supervised\<RUN_ID>\
    ├── registry\                 # JSONL registry + split JSON
    ├── features\                 # Day 13 feature cache
    ├── checkpoints\              # Day 15 model checkpoints
    └── reports\                  # Metrics, curves, failure cases
```

## Quick-start (Day 11B pilot validation)

From the repo root (`asv_vla_pc\repo`):

```powershell
# 1. Environment
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pytest jsonschema pillow numpy pyyaml

# 2. Copy pilot data into place
New-Item -ItemType Directory -Force data\incoming, data\extracted\pilot | Out-Null
Copy-Item <path-to>\day10_A1D7BAAE49F39E3BB7B1808AB8443CA9.tar.gz data\incoming\
tar -xzf data\incoming\day10_A1D7BAAE49F39E3BB7B1808AB8443CA9.tar.gz -C data\extracted\pilot

# 3. Verify SHA-256 (compare with the value recorded in ROADMAP_RESUME.md)
certutil -hashfile data\incoming\day10_A1D7BAAE49F39E3BB7B1808AB8443CA9.tar.gz SHA256

# 4. Verify supervision data
$env:PYTHONPATH = "src\asv_vla"
python -c "from asv_vla.supervised_dataset import evaluate_main; raise SystemExit(evaluate_main())" data\extracted\pilot\artifacts\day10_supervised\A1D7BAAE49F39E3BB7B1808AB8443CA9 --require-all-labels
```

## Build registry

```powershell
$env:PYTHONPATH = "src\asv_vla"
python -m training.dataset_registry --data-root data\extracted\pilot --output data\registry\dataset_registry_v1.jsonl
```

**Expected output for the old 50-frame pilot:**
```text
DAY11_REGISTRY_PASS ... eligible_runs=0 training_ready=False
DAY11_TRAINING_NOT_READY: need at least 12 eligible Runs ...
```

## Create splits

```powershell
python -m training.make_group_splits --registry data\registry\dataset_registry_v1.jsonl --output data\registry\group_split_v1.json
```

**Expected output for pilot:**
```text
DAY11_SPLIT_PASS runs=0 seeds=0 train=0 val=0 test=0 training_ready=False
DAY11_TRAINING_NOT_READY: registry has no training-eligible Runs
```

## Run tests

```powershell
$env:PYTHONPATH = "src\asv_vla"
python -m pytest -q training\test
```

## When `training_ready` becomes `true`

An eligible Run must have at least 80 frames, a passing quality report,
complete 9/9 supervision, and Day 12 collection-slot metadata. The registry
requires at least 12 eligible Runs across at least three Scene Seeds.
The 12-Run split is frozen to 8/2/2; the 30-Run split is 18/6/6.

Use `training/day12_collection.py` to verify that the recorded entity
geometry really matches the counterbalanced plan. Full instructions are in
`docs/DAY12_COLLECTION.md`.

## Environment report

Before any training, capture and save this information:

```powershell
# Windows
systeminfo | findstr /B /C:"OS Name" /C:"OS Version"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA {torch.version.cuda}')"
```

Save the output to `data\reports\environment_v1.json`.
