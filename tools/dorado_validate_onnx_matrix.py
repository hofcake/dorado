#!/usr/bin/env python3
"""
Run ONNX validation across a curated Dorado model matrix.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dorado_validate_onnx import validate_model


DEFAULT_MODELS = [
    "dna_r9.4.1_e8_fast@v3.4",
    "dna_r9.4.1_e8_hac@v3.3",
    "dna_r9.4.1_e8_sup@v3.6",
    "dna_r10.4.1_e8.2_260bps_hac@v3.5.2",
    "dna_r10.4.1_e8.2_400bps_hac@v4.0.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.1.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
    "dna_r10.4.1_e8.2_400bps_hac@v4.3.0",
    "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
    "dna_r10.4.1_e8.2_400bps_hac@v5.2.0",
    "dna_r10.4.1_e8.2_400bps_fast@v5.2.0",
    "dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
    "dna_r10.4.1_e8.2_400bps_sup@v5.0.0",
    "rna002_70bps_hac@v3",
    "rna004_130bps_fast@v5.3.0",
    "rna004_130bps_hac@v3.0.1",
    "rna004_130bps_hac@v5.3.0",
    "rna004_130bps_sup@v5.3.0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ONNX export accuracy across a curated Dorado model matrix."
    )
    parser.add_argument(
        "--models",
        default="",
        help="comma-separated model names; defaults to a curated cross-version matrix",
    )
    parser.add_argument(
        "--models-directory",
        default="models",
        help="directory used to cache downloaded models",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory receiving per-model validation reports and the matrix summary",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="",
        help="comma-separated chunk sizes to validate; defaults to three aligned sizes per model",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2",
        help="comma-separated batch sizes to validate",
    )
    parser.add_argument(
        "--seeds",
        default="1234",
        help="comma-separated random seeds used for generated validation inputs",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to use if exporting is needed",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="relative tolerance for output comparison",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.5e-3,
        help="absolute tolerance for output comparison; older LSTM exports show small kernel-level drift around 1e-3",
    )
    parser.add_argument(
        "--reexport",
        action="store_true",
        help="force a fresh ONNX export for each model",
    )
    return parser.parse_args()


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()] or DEFAULT_MODELS
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_reports = []
    for model_name in models:
        report = validate_model(
            model_arg=model_name,
            models_directory=Path(args.models_directory),
            output_dir=output_dir / slugify_model_name(model_name),
            chunk_sizes_arg=args.chunk_sizes,
            batch_sizes_arg=args.batch_sizes,
            seeds_arg=args.seeds,
            opset=args.opset,
            rtol=args.rtol,
            atol=args.atol,
            reexport=args.reexport,
        )
        model_reports.append(
            {
                "model_name": report["model_name"],
                "architecture": report["architecture"],
                "passed": report["passed"],
                "num_cases": report["summary"]["num_cases"],
                "max_abs_diff": report["summary"]["max_abs_diff"],
                "max_rel_diff": report["summary"]["max_rel_diff"],
                "report_path": str(
                    (output_dir / slugify_model_name(model_name) / "validation_report.json")
                ),
            }
        )

    summary = {
        "models": model_reports,
        "num_models": len(model_reports),
        "num_passed": sum(1 for item in model_reports if item["passed"]),
        "num_failed": sum(1 for item in model_reports if not item["passed"]),
        "overall_passed": all(item["passed"] for item in model_reports),
        "max_abs_diff": max(item["max_abs_diff"] for item in model_reports),
        "max_rel_diff": max(item["max_rel_diff"] for item in model_reports),
    }
    summary_path = output_dir / "validation_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not summary["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
