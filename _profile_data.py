"""Temporary dataset profiler - deleted after inspection."""
from __future__ import annotations

import csv
import glob
import os
import sys

csv.field_size_limit(10_000_000)

for path in sorted(glob.glob("data/raw/*.csv")):
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = 0
            samples = []
            for row in reader:
                rows += 1
                if rows <= 2:
                    samples.append(row)
    except Exception as error:
        print(f"{name}: FAILED {error}")
        continue
    print(f"--- {name}  ({size_mb:.1f} MB, {rows} data rows)")
    print(f"    columns: {header}")
    for row in samples:
        preview = [value[:60].replace("\n", " ") for value in row[: len(header)]]
        print(f"    row: {preview}")
    sys.stdout.flush()
