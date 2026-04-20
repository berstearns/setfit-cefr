"""Inference pipeline: load a trained model folder and score a list of CSVs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from setfit_cefr.config import Config
from setfit_cefr.data import load_test_file
from setfit_cefr.hashing import file_fingerprint, predict_hash
from setfit_cefr.reporting import compute_metrics, render_markdown_report, write_report

log = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_model(model_dir: Path, device: str):
    from setfit import SetFitModel

    model_subdir = model_dir / "setfit-model"
    if not model_subdir.exists():
        raise FileNotFoundError(
            f"Expected trained SetFit model at {model_subdir}; "
            f"did you run train.py against this folder?"
        )
    model = SetFitModel.from_pretrained(str(model_subdir), device=device)
    return model


def predict(
    model_dir: str | Path,
    test_files: list[str | Path],
    config: Config,
    predict_overrides_yaml: str = "",
) -> Path:
    """Score each test file, write predictions + metrics, return the output folder."""
    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    # 1. Identity & paths.
    model_hash_str = (model_dir / "config.hash").read_text().strip() if (
        model_dir / "config.hash"
    ).exists() else model_dir.name
    labels_path = model_dir / "labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
    else:
        labels = config.data.label_order

    p_hash = predict_hash(model_hash_str, test_files, predict_overrides_yaml)
    out_dir = Path(config.runtime.predictions_root) / p_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Prediction hash %s -> %s", p_hash, out_dir)

    # 2. Load model.
    device = _resolve_device(config.runtime.device)
    model = _load_model(model_dir, device=device)

    # 3. Score each test file.
    probas_payload: dict = {
        "model_hash": model_hash_str,
        "model_path": str(model_dir),
        "predict_hash": p_hash,
        "labels": labels,
        "test_files": {},
    }
    report: dict = {
        "model_hash": model_hash_str,
        "model_path": str(model_dir),
        "predict_hash": p_hash,
        "labels": labels,
        "test_files": {},
    }

    test_fingerprints = []
    for f in test_files:
        p = Path(f)
        name = p.name
        log.info("Scoring %s", p)
        df = load_test_file(p, config.data)

        texts = df[config.data.text_column].tolist()
        if not texts:
            log.warning("No rows to score in %s", p)

        probas = _predict_proba(model, texts, labels)
        pred_idx = np.argmax(probas, axis=1) if len(texts) else np.array([], dtype=int)
        pred_labels = [labels[i] for i in pred_idx.tolist()]

        has_gold = config.data.label_column in df.columns
        y_true: list[str] = (
            df[config.data.label_column].astype(str).tolist() if has_gold else []
        )

        # Numpy arrays -> plain lists for JSON.
        ids = (
            df[config.data.id_column].astype(str).tolist()
            if config.data.id_column in df.columns
            else [str(i) for i in range(len(df))]
        )

        probas_payload["test_files"][name] = {
            "path": str(p),
            "n_samples": int(len(df)),
            "ids": ids,
            "y_true": y_true,
            "y_pred": pred_labels,
            "probas": probas.tolist(),
        }

        metrics = compute_metrics(y_true, pred_labels, labels, config.reporting)
        report["test_files"][name] = {
            "path": str(p),
            "n_samples": int(len(df)),
            "metrics": metrics,
        }
        test_fingerprints.append(file_fingerprint(p))

    # 4. Persist artefacts.
    (out_dir / "predictions_probas.json").write_text(
        json.dumps(probas_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(report, out_dir)
    (out_dir / "report.md").write_text(
        render_markdown_report(report, model_hash_str, str(model_dir), labels),
        encoding="utf-8",
    )
    (out_dir / "predict_config.yaml").write_text(config.to_yaml(), encoding="utf-8")
    (out_dir / "predict_manifest.json").write_text(
        json.dumps(
            {
                "model_hash": model_hash_str,
                "model_path": str(model_dir),
                "test_fingerprints": test_fingerprints,
                "predict_overrides_yaml": predict_overrides_yaml,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Predictions written to %s", out_dir)
    return out_dir


def _predict_proba(model, texts: list[str], labels: list[str]) -> np.ndarray:
    """Robust ``predict_proba`` wrapper that always returns ``(n, num_labels)``.

    SetFit's scikit-learn head supports ``predict_proba`` directly. The
    differentiable head returns logits; we softmax if needed.
    """
    if not texts:
        return np.zeros((0, len(labels)), dtype=np.float32)
    proba = model.predict_proba(texts, as_numpy=True)
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] != len(labels):
        raise RuntimeError(
            f"predict_proba returned {arr.shape[1]} columns; expected {len(labels)} "
            f"for labels {labels}"
        )
    # Normalise in case the head returned logits.
    row_sums = arr.sum(axis=1, keepdims=True)
    looks_like_probas = np.allclose(row_sums, 1.0, atol=1e-3) and (arr >= 0).all()
    if not looks_like_probas:
        arr = np.exp(arr - arr.max(axis=1, keepdims=True))
        arr = arr / arr.sum(axis=1, keepdims=True)
    return arr.astype(np.float32)
