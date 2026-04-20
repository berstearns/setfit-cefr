"""Metrics computation and human-readable report rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from setfit_cefr.config import ReportingConfig


def _confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return m


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return float("nan")
    return sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    f1s: list[float] = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else float("nan")


def _quadratic_weighted_kappa(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> float:
    """Cohen's kappa with quadratic penalty — the standard metric for CEFR."""
    if not y_true:
        return float("nan")
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    obs = np.zeros((n, n), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            obs[idx[t], idx[p]] += 1
    if obs.sum() == 0:
        return float("nan")
    obs /= obs.sum()
    hist_t = obs.sum(axis=1)
    hist_p = obs.sum(axis=0)
    exp = np.outer(hist_t, hist_p)
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            w[i, j] = ((i - j) ** 2) / ((n - 1) ** 2) if n > 1 else 0.0
    num = (w * obs).sum()
    den = (w * exp).sum()
    return float(1.0 - num / den) if den > 0 else float("nan")


def _adjacent_accuracy(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    if not y_true:
        return float("nan")
    idx = {lab: i for i, lab in enumerate(labels)}
    hits = 0
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx and abs(idx[t] - idx[p]) <= 1:
            hits += 1
    return hits / len(y_true)


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    cfg: ReportingConfig,
) -> dict[str, Any]:
    """Compute the full metric bundle. Gracefully skips when labels are missing."""
    if not y_true:
        return {"n": 0, "note": "no ground-truth labels available"}

    out: dict[str, Any] = {
        "n": len(y_true),
        "accuracy": _accuracy(y_true, y_pred),
    }
    if cfg.compute_macro_f1:
        out["macro_f1"] = _macro_f1(y_true, y_pred, labels)
    if cfg.compute_qwk:
        out["quadratic_weighted_kappa"] = _quadratic_weighted_kappa(y_true, y_pred, labels)
    if cfg.compute_adjacent_accuracy:
        out["adjacent_accuracy"] = _adjacent_accuracy(y_true, y_pred, labels)
    if cfg.save_confusion_matrix:
        out["confusion_matrix"] = {
            "labels": labels,
            "matrix": _confusion_matrix(y_true, y_pred, labels),
        }
    return out


def render_markdown_report(
    report: dict[str, Any],
    model_hash: str,
    model_path: str,
    labels: list[str],
) -> str:
    """Render the JSON report as a single Markdown document."""
    lines: list[str] = []
    lines.append(f"# Prediction report — model `{model_hash}`")
    lines.append("")
    lines.append(f"- Model folder: `{model_path}`")
    lines.append(f"- Label order: `{labels}`")
    lines.append(f"- Test files: {len(report.get('test_files', {}))}")
    lines.append("")

    for name, entry in report.get("test_files", {}).items():
        lines.append(f"## `{name}`")
        lines.append("")
        lines.append(f"- Path: `{entry.get('path')}`")
        lines.append(f"- Samples: **{entry.get('n_samples')}**")
        metrics = entry.get("metrics", {})
        if metrics.get("n", 0) == 0:
            lines.append(f"- Metrics: _{metrics.get('note', 'skipped')}_")
        else:
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for key in (
                "accuracy",
                "macro_f1",
                "quadratic_weighted_kappa",
                "adjacent_accuracy",
            ):
                if key in metrics:
                    lines.append(f"| {key} | {metrics[key]:.4f} |")
            cm = metrics.get("confusion_matrix")
            if cm:
                lines.append("")
                lines.append("### Confusion matrix (rows = true, cols = pred)")
                lines.append("")
                header = "| true \\ pred | " + " | ".join(cm["labels"]) + " |"
                sep = "|" + "---|" * (len(cm["labels"]) + 1)
                lines.append(header)
                lines.append(sep)
                for lab, row in zip(cm["labels"], cm["matrix"]):
                    lines.append("| " + lab + " | " + " | ".join(str(v) for v in row) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
