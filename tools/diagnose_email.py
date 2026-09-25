"""Inspect tokenization and raw model output for one email.

Examples:
  python -m tools.diagnose_email --subject "Prize offer" --body "..."
  python -m tools.diagnose_email --file message.eml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.email_spam.preprocessing import normalize_email, parse_uploaded_file
from src.email_spam.registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose one email through the production model pipeline.")
    parser.add_argument("--subject", default="", help="Email subject")
    parser.add_argument("--body", default="", help="Email body")
    parser.add_argument("--file", type=Path, help=".eml or .txt file to inspect")
    parser.add_argument("--artifacts", default="models")
    args = parser.parse_args()

    if args.file:
        subject, body = parse_uploaded_file(args.file.name, args.file.read_bytes())
    else:
        subject, body = args.subject, args.body
    if not subject.strip() and not body.strip():
        parser.error("provide --subject/--body or --file")

    email = normalize_email(subject, body)
    registry = ModelRegistry.instance(args.artifacts)
    encoded = registry.encode([email.combined_text])
    token_ids = encoded[0]
    oov_id = int(registry.tokenizer.word_index.get(registry.tokenizer.oov_token, 1))
    non_padding = token_ids[token_ids != 0]
    oov_count = int((non_padding == oov_id).sum())
    prediction = registry.predict_text(email.combined_text)

    print(json.dumps({
        "subject": email.subject,
        "normalized_text": email.combined_text,
        "token_ids": [int(value) for value in token_ids],
        "non_padding_tokens": int(len(non_padding)),
        "oov_token_id": oov_id,
        "oov_tokens": oov_count,
        "oov_rate": float(oov_count / max(len(non_padding), 1)),
        "padded_sequence_length": int(len(token_ids)),
        "raw_probability_spam": prediction.probability,
        "prediction": prediction.label,
        "threshold": registry.threshold,
    }, indent=2))


if __name__ == "__main__":
    main()
