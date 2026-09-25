"""Fit temperature scaling on a held-out split.

Example:
  python -m tools.calibrate --data data/raw/CEAS_08.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.email_spam.calibration import calibration_report, save_calibration
from src.email_spam.data import load_email_csv, split_dataset
from src.email_spam.registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Create models/calibration.json from held-out predictions.")
    parser.add_argument("--data", required=True, nargs="+", help="CSV files used for the model")
    parser.add_argument("--artifacts", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=128, help="Prediction batch size during calibration")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    import pandas as pd

    frames = [load_email_csv(path, require_both_labels=False) for path in args.data]
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    train, validation, test = split_dataset(frame, args.seed)
    held_out = validation if args.split == "validation" else test

    registry = ModelRegistry.instance(args.artifacts)
    texts = held_out["text"].tolist()
    probabilities = np.concatenate([
        np.asarray(
            [prediction.probability for prediction in registry.predict_batch(texts[index:index + args.batch_size]).predictions],
            dtype=float,
        )
        for index in range(0, len(texts), args.batch_size)
    ])
    labels = held_out["label"].to_numpy(dtype=int)
    report = calibration_report(probabilities, labels, threshold=registry.threshold)
    report["dataset"] = [str(Path(path).name) for path in args.data]
    report["split"] = args.split
    report["seed"] = args.seed
    save_calibration(report, Path(args.artifacts) / "calibration.json")
    print(json.dumps({
        "path": str(Path(args.artifacts) / "calibration.json"),
        "temperature": report.get("temperature"),
        "samples": report.get("samples"),
        "before": report.get("before"),
        "after": report.get("after"),
    }, indent=2))


if __name__ == "__main__":
    main()
