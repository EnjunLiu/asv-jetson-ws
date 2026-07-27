from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .language_intervention_dataset import (
    LanguageDatasetError,
    default_dataset_dir,
    load_and_validate,
)


def parse_args(argv=None):
    dataset_dir = default_dataset_dir()
    parser = argparse.ArgumentParser(
        description="Validate paired language interventions and report coverage."
    )
    parser.add_argument(
        "--instructions",
        default=str(dataset_dir / "instructions.jsonl"),
    )
    parser.add_argument(
        "--pairs",
        default=str(dataset_dir / "contrast_pairs.jsonl"),
    )
    parser.add_argument(
        "--output",
        help="Optional JSON report path.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        _, _, report = load_and_validate(
            args.instructions,
            args.pairs,
        )
    except LanguageDatasetError as exc:
        print(f"LANGUAGE_INTERVENTION_COVERAGE_FAIL:{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    report_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    print(report_text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text + "\n", encoding="utf-8")
        print(f"report={output_path}")
    print("LANGUAGE_INTERVENTION_COVERAGE_PASS")


if __name__ == "__main__":
    main()
