# SignalGuard Email Intelligence

SignalGuard is a production shaped deep learning email spam classifier. 

## What it does

- Accepts an email subject and body, or a `.eml` / `.txt` file.
- Removes HTML for analysis without executing scripts, resources, or attachments.
- Normalizes URLs, addresses, phone numbers, Unicode, repeated characters, and whitespace while preserving useful spam signals.
- Trains an LSTM using train only tokenizer fitting, stratified train/validation/test splits, and balanced class weights.
- Saves the model, tokenizer, preprocessing metadata, training history, and real evaluation metrics.
- Serves a responsive dashboard with prediction, spam probability, confidence, risk, model notes, and session history.

## Architecture

`CSV -> normalized subject/body -> tokenizer -> padded sequences -> Embedding -> SpatialDropout -> LSTM -> Dense -> sigmoid`

The explanation panel contains deterministic supporting signals and explicitly labels the trained model probability. These are not fabricated token attributions or SHAP values; reliable gradient attribution was not added to this LSTM pipeline.

## Dataset format

Put a genuine email dataset at `data/raw/emails.csv`. Required columns are `label` and one of `body`, `text`, `message`, or `content`; `subject` is optional. Labels can be `spam`/`ham`, `spam`/`legitimate`, or `1`/`0`. SMS data is not equivalent to email data and must not be used to claim email performance.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train and evaluate

```powershell
python train.py --data data/raw/emails.csv --epochs 8
```

This creates `models/email_spam_lstm.keras`, `tokenizer.json`, `metadata.json`, `evaluation.json`, and `history.json`. The evaluation file contains accuracy, precision, recall, F1, ROC-AUC, PR-AUC, false-positive rate, false negative rate, and the confusion matrix.

## Run the application

```powershell
python -m app.main
```

Open `http://127.0.0.1:5000`. The app reports that the model is unavailable until training artifacts exist; it never invents metrics or predictions.

## Diagnose and calibrate

Inspect the exact normalized text, token IDs, OOV rate, padding length, and raw score for one message:

```powershell
python -m tools.diagnose_email --subject "Limited offer" --body "Click here to claim your reward: https://example.com"
```

Fit temperature scaling on the held out validation split and create `models/calibration.json`:

```powershell
python -m tools.calibrate --data data/raw/CEAS_08.csv --split validation --batch-size 64
```

Restart the Flask server after calibration so the in-memory model registry loads the new artifact. The analyzer will then show calibrated confidence when the artifact is available.

## Tests

```powershell
pytest
```

## Project structure

`src/email_spam/` contains preprocessing, dataset loading, model construction, metrics, and inference. `train.py` is deliberately separate from `app/main.py`. `app/` contains the Flask API and dashboard. `tests/` covers normalization, label mapping, `.eml` parsing, invalid uploads, and metric bounds.

## Limitations and future work

Performance depends on the quality, recency and domain coverage of the supplied email dataset. The default threshold is 0.5 and should be calibrated on a validation set when false positives are especially costly. Future improvements include a larger pretrained email language encoder, threshold calibration, persistent authenticated history, drift monitoring, and model specific integrated gradients attribution.
