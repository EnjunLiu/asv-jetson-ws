# Language intervention dataset

This directory contains paired Chinese instructions for evaluating whether the
trajectory policy responds to language changes in an otherwise fixed scene.
The labels organize data and evaluation only; they are not outputs of an online
task parser.

## Files

- `instructions.jsonl`: instruction text, intent group, target attribute,
  distance bucket, split, and template family.
- `contrast_pairs.jsonl`: two conflicting instructions assigned to the same
  `scene_seed`, plus the intervention type and expected trajectory effect.

## Reproduce and validate

From the workspace root:

```bash
source .venv/bin/activate
PYTHONPATH=src/asv_vla python -m asv_vla.generate_language_interventions
PYTHONPATH=src/asv_vla python -m asv_vla.evaluate_language_coverage
```

The checked-in files contain 90 instructions and 24 contrast pairs. Training,
validation, and test splits use disjoint template families.
