"""CLI entrypoint to execute Phase 8 accuracy and KPI evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from .phase8_accuracy import run_phase8_evaluations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase8"),
        help="Directory where evaluation artifacts will be written.",
    )
    return parser.parse_args()


def main() -> Dict[str, object]:
    args = _parse_args()
    results = run_phase8_evaluations(args.output)
    summary = {domain: evaluation.to_dict() for domain, evaluation in results.items()}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
