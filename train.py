"""Train the email model from one or more compatible CSV files.

Over-confidence controls, all opt-in and reported in ``metadata.json``:

* ``--label-smoothing`` softens the hard 0/1 targets (default 0.05).
* ``--l2`` adds L2 weight decay to the LSTM and dense kernels.
* ``--dense-dropout`` inserts dropout before the output unit.
* ``--early-stopping-patience`` stops when validation loss stops improving and
  restores the best epoch, so training loss can no longer diverge from
  validation loss unnoticed.
* Temperature scaling is fitted on the held-out validation split after
  training and written to ``calibration.json`` so the dashboard can show a
  calibrated probability instead of the raw over-confident sigmoid.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.email_spam.calibration import calibration_report, evaluate_temperature, save_calibration
from src.email_spam.data import load_email_csv, split_dataset
from src.email_spam.metrics import calculate_metrics, save_metrics
from src.email_spam.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LSTM email spam classifier.")
    parser.add_argument("--data", required=True, nargs="+", help="CSV files containing subject/body/text and label columns")
    parser.add_argument("--artifacts", default="models")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability cutoff for SPAM decisions")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Soften 0/1 targets (0 disables)")
    parser.add_argument("--l2", type=float, default=1e-5, help="L2 weight decay on LSTM/dense kernels (0 disables)")
    parser.add_argument("--dense-dropout", type=float, default=0.2, help="Dropout before the output unit (0 disables)")
    parser.add_argument("--early-stopping-patience", type=int, default=2, help="Epochs without val_loss improvement before stopping (0 disables)")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum val_loss improvement counted by early stopping")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip fitting temperature scaling after training")
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    if not 0.0 <= args.label_smoothing < 0.5:
        parser.error("--label-smoothing must be in [0, 0.5)")
    if not 0.0 <= args.l2:
        parser.error("--l2 must be non-negative")
    if not 0.0 <= args.dense_dropout < 1.0:
        parser.error("--dense-dropout must be in [0, 1)")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    frames = [load_email_csv(path, require_both_labels=False) for path in args.data]
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if frame["label"].nunique() < 2:
        raise ValueError("The combined training data must contain both spam and legitimate examples.")
    train, validation, test = split_dataset(frame, args.seed)
    vocabulary_size, sequence_length = 20_000, 300
    tokenizer = Tokenizer(num_words=vocabulary_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train["text"])

    def encode(values):
        return pad_sequences(tokenizer.texts_to_sequences(values), maxlen=sequence_length, padding="post", truncating="post")

    x_train, x_validation, x_test = map(encode, (train["text"], validation["text"], test["text"]))
    y_train, y_validation, y_test = train["label"].values, validation["label"].values, test["label"].values
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    model = build_model(
        vocabulary_size,
        sequence_length,
        l2_strength=args.l2,
        dense_dropout=args.dense_dropout,
    )
    loss = BinaryCrossentropy(label_smoothing=args.label_smoothing)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                min_delta=args.min_delta,
                restore_best_weights=True,
                verbose=1,
            )
        )
    history = model.fit(
        x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,
        batch_size=64, class_weight={0: float(weights[0]), 1: float(weights[1])},
        callbacks=callbacks, verbose=1,
    )
    probabilities = model.predict(x_test, verbose=0).reshape(-1)
    metrics = calculate_metrics(y_test, probabilities)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    model.save(artifacts / "email_spam_lstm.keras")
    (artifacts / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")

    calibration_payload = None
    if not args.no_calibrate:
        validation_probabilities = model.predict(x_validation, verbose=0).reshape(-1)
        calibration_payload = calibration_report(validation_probabilities, y_validation, threshold=args.threshold)
        calibration_payload["dataset"] = [str(Path(path).name) for path in args.data]
        calibration_payload["split"] = "validation"
        calibration_payload["seed"] = args.seed
        # The validation split selects both early-stopping weights and the
        # temperature. Report the final calibration quality on untouched test
        # predictions, without using those labels to alter the model.
        calibration_payload["test_evaluation"] = evaluate_temperature(
            probabilities, y_test, calibration_payload["temperature"]
        )
        save_calibration(calibration_payload, artifacts / "calibration.json")

    epochs_run = len(history.history.get("loss", []))
    (artifacts / "metadata.json").write_text(json.dumps({
        "architecture": (
            f"Embedding(96) -> SpatialDropout1D -> LSTM(64) -> Dense(32) -> "
            f"Dense(1, sigmoid)"
        ),
        "padding_masked": True,
        "sequence_length": sequence_length, "vocabulary_size": vocabulary_size, "threshold": args.threshold,
        "l2_strength": args.l2, "label_smoothing": args.label_smoothing, "dense_dropout": args.dense_dropout,
        "early_stopping_patience": args.early_stopping_patience,
        "epochs_requested": args.epochs, "epochs_run": epochs_run,
        "temperature": (calibration_payload or {}).get("temperature"),
        "dataset": [str(Path(path).name) for path in args.data], "seed": args.seed, "train_count": len(train),
        "validation_count": len(validation), "test_count": len(test),
    }, indent=2), encoding="utf-8")
    save_metrics(metrics, artifacts / "evaluation.json")
    (artifacts / "history.json").write_text(json.dumps({key: [float(value) for value in values] for key, values in history.history.items()}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
