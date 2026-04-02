#!/usr/bin/env python3
"""
Validate Dorado ONNX exports against the reconstructed PyTorch model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from dorado_export_onnx import (
    build_model,
    export_model,
    load_model_config,
    load_model_weights,
    load_weights_into_model,
    resolve_model_path,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Dorado ONNX export against the reconstructed PyTorch model."
    )
    parser.add_argument("--model", required=True, help="Dorado model name or local model directory")
    parser.add_argument(
        "--models-directory",
        default="models",
        help="directory used to cache downloaded models when --model is a model name",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory containing or receiving the ONNX export and validation report",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="",
        help="comma-separated chunk sizes to validate; defaults to three aligned sizes",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2",
        help="comma-separated batch sizes to validate against the ONNX dynamic batch axis",
    )
    parser.add_argument(
        "--seeds",
        default="1234,4321",
        help="comma-separated RNG seeds used to generate validation inputs",
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
        "--seed",
        type=int,
        default=1234,
        help="random seed for generated validation inputs",
    )
    parser.add_argument(
        "--reexport",
        action="store_true",
        help="force a fresh ONNX export before validation",
    )
    return parser.parse_args()


def parse_positive_ints(arg: str, label: str) -> list[int]:
    values = []
    for item in arg.split(","):
        value = int(item.strip())
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"No {label} values were provided")
    return values


def parse_chunk_sizes(arg: str, normalized: int, granularity: int) -> list[int]:
    if arg:
        values = parse_positive_ints(arg, "chunk size")
        for value in values:
            if value % granularity != 0:
                raise ValueError(
                    f"Chunk size {value} is not aligned to model granularity {granularity}"
                )
        return values
    return [normalized, normalized + granularity, normalized + (2 * granularity)]


def build_runtime_session(onnx_path: Path) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def compare_outputs(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    config_chunk_size: int,
    num_features: int,
    batch_size: int,
    seed: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_tensor = torch.randn(batch_size, num_features, config_chunk_size, dtype=torch.float32)

    with torch.inference_mode():
        pytorch_out = model(input_tensor).cpu().numpy()

    onnx_out = session.run(["scores"], {"input": input_tensor.numpy()})[0]

    abs_diff = np.abs(pytorch_out - onnx_out)
    denom = np.maximum(np.abs(pytorch_out), 1e-12)
    rel_diff = abs_diff / denom

    return {
        "batch_size": batch_size,
        "chunk_size": config_chunk_size,
        "shape": list(pytorch_out.shape),
        "max_abs_diff": float(abs_diff.max(initial=0.0)),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_rel_diff": float(rel_diff.max(initial=0.0)),
        "mean_rel_diff": float(rel_diff.mean()),
        "allclose": bool(np.allclose(pytorch_out, onnx_out, atol=atol, rtol=rtol)),
    }


def validate_model(
    model_arg: str,
    models_directory: Path,
    output_dir: Path,
    chunk_sizes_arg: str,
    batch_sizes_arg: str,
    seeds_arg: str,
    opset: int,
    rtol: float,
    atol: float,
    reexport: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = resolve_model_path(model_arg, models_directory, False)
    config = load_model_config(model_dir)
    weights = load_model_weights(config)
    model = build_model(config)
    load_weights_into_model(model, weights)
    model.eval()

    normalized_chunk = config.normalized_chunk_size()
    chunk_sizes = parse_chunk_sizes(
        chunk_sizes_arg, normalized_chunk, config.chunk_size_granularity()
    )
    batch_sizes = parse_positive_ints(batch_sizes_arg, "batch size")
    seeds = parse_positive_ints(seeds_arg, "seed")

    onnx_path = output_dir / "model.onnx"
    if reexport or not onnx_path.exists():
        export_model(model, config, output_dir, "model.onnx", opset, chunk_sizes[0])
        write_manifest(output_dir, config, onnx_path, chunk_sizes[0], opset)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    session = build_runtime_session(onnx_path)
    results = []
    for seed in seeds:
        for batch_size in batch_sizes:
            for chunk_size in chunk_sizes:
                results.append(
                    compare_outputs(
                        model=model,
                        session=session,
                        config_chunk_size=chunk_size,
                        num_features=config.num_features,
                        batch_size=batch_size,
                        seed=seed,
                        atol=atol,
                        rtol=rtol,
                    )
                )

    passed = all(item["allclose"] for item in results)

    report = {
        "model_name": config.model_name,
        "model_path": str(model_dir),
        "onnx_path": str(onnx_path),
        "architecture": "transformer" if config.is_tx else "crf",
        "chunk_sizes": chunk_sizes,
        "batch_sizes": batch_sizes,
        "seeds": seeds,
        "thresholds": {"atol": atol, "rtol": rtol},
        "results": results,
        "passed": passed,
        "summary": {
            "num_cases": len(results),
            "max_abs_diff": max(item["max_abs_diff"] for item in results),
            "max_rel_diff": max(item["max_rel_diff"] for item in results),
            "mean_abs_diff": float(np.mean([item["mean_abs_diff"] for item in results])),
            "mean_rel_diff": float(np.mean([item["mean_rel_diff"] for item in results])),
        },
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = validate_model(
        model_arg=args.model,
        models_directory=Path(args.models_directory),
        output_dir=Path(args.output_dir).resolve(),
        chunk_sizes_arg=args.chunk_sizes,
        batch_sizes_arg=args.batch_sizes,
        seeds_arg=args.seeds,
        opset=args.opset,
        rtol=args.rtol,
        atol=args.atol,
        reexport=args.reexport,
    )

    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
