"""Shared application state.

The trained network is created once, lazily, and reused for the lifetime of the
process. No route ever constructs a model, reads a dataset, or triggers training.
"""

from __future__ import annotations

from pathlib import Path

from src.email_spam.history import HistoryStore
from src.email_spam.registry import ARTIFACT_FILES, OPTIONAL_ARTIFACTS, ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models"

history = HistoryStore()


def model_available() -> bool:
    """True only when every required artifact is present on disk."""
    return all((ARTIFACT_DIR / name).exists() for name in ARTIFACT_FILES.values())


def get_registry() -> ModelRegistry:
    """Return the process-wide registry, loading artifacts on first use only."""
    return ModelRegistry.instance(ARTIFACT_DIR)


def optional_artifact(key: str) -> dict | None:
    """Read a precomputed experiment artifact, or None when it is absent."""
    return get_registry().auxiliary.get(key)


def artifact_inventory() -> dict:
    """Describe every artifact file present or absent, without loading the model."""
    inventory: dict[str, dict] = {}
    for key, name in {**ARTIFACT_FILES, **OPTIONAL_ARTIFACTS}.items():
        path = ARTIFACT_DIR / name
        inventory[key] = {
            "file": name,
            "present": path.exists(),
            "bytes": int(path.stat().st_size) if path.exists() else 0,
        }
    return inventory
