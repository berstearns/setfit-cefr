from __future__ import annotations

from setfit_cefr.config import ReportingConfig
from setfit_cefr.reporting import compute_metrics, render_markdown_report

LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def test_compute_metrics_all_correct():
    y = ["A1", "B1", "C2"]
    m = compute_metrics(y, y, LABELS, ReportingConfig())
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] > 0
    assert m["adjacent_accuracy"] == 1.0


def test_compute_metrics_adjacent_counts_off_by_one():
    y_t = ["A1", "B1"]
    y_p = ["A2", "B2"]  # each off by exactly one level
    m = compute_metrics(y_t, y_p, LABELS, ReportingConfig())
    assert m["accuracy"] == 0.0
    assert m["adjacent_accuracy"] == 1.0


def test_compute_metrics_handles_empty():
    m = compute_metrics([], [], LABELS, ReportingConfig())
    assert m == {"n": 0, "note": "no ground-truth labels available"}


def test_render_markdown_smoke():
    report = {
        "test_files": {
            "foo.csv": {
                "path": "/tmp/foo.csv",
                "n_samples": 2,
                "metrics": {
                    "n": 2,
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                    "quadratic_weighted_kappa": 0.3,
                    "adjacent_accuracy": 1.0,
                    "confusion_matrix": {
                        "labels": ["A1", "A2"],
                        "matrix": [[1, 0], [0, 1]],
                    },
                },
            }
        }
    }
    md = render_markdown_report(report, "deadbeef1234", "models/deadbeef1234", ["A1", "A2"])
    assert "deadbeef1234" in md
    assert "foo.csv" in md
    assert "| accuracy | 0.5000 |" in md
