"""Deterministic content-addressable identifiers for models and predictions.

We hash the canonical YAML of a resolved config so that two runs with the same
effective settings land in the same ``models/<hash>/`` folder and two
prediction runs with the same (model, inputs, override) triple collide in
``predictions/<hash>/``.

Hashes are truncated to 12 hex chars — enough to avoid collisions in practice
while staying short enough for humans to type.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from setfit_cefr.config import Config, canonical_yaml

HASH_LEN = 12


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def model_hash(config: Config) -> str:
    """Hash the canonical YAML of the resolved training config."""
    return _sha256(canonical_yaml(config))[:HASH_LEN]


def predict_hash(
    model_hash_str: str,
    test_files: Iterable[str | Path],
    predict_overrides_yaml: str = "",
) -> str:
    """Hash the (model, sorted test paths, overrides) triple."""
    sorted_files = sorted(str(Path(p).resolve()) for p in test_files)
    parts = [model_hash_str, *sorted_files, predict_overrides_yaml]
    blob = "\n".join(parts)
    return _sha256(blob)[:HASH_LEN]


def file_fingerprint(path: str | Path) -> dict[str, str | int]:
    """Record size + sha256 of a data file, for manifests.

    We hash the raw bytes so identical copies in different folders collapse
    and mismatched ones (say, someone resampled the test set) don't silently
    reuse old predictions.
    """
    p = Path(path)
    h = hashlib.sha256()
    size = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return {"path": str(p), "size_bytes": size, "sha256": h.hexdigest()}
