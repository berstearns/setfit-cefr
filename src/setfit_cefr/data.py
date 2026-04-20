"""Data loading, cleaning, sampling, and train/eval splitting."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from setfit_cefr.config import DataConfig

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("text_column", "label_column", "id_column")


class DataError(RuntimeError):
    """Raised when inputs violate the schema required by the config."""


def load_and_concat(files: Iterable[str | Path], cfg: DataConfig) -> pd.DataFrame:
    """Load all training CSVs, validate columns, stack them."""
    frames: list[pd.DataFrame] = []
    for f in files:
        p = Path(f)
        if not p.exists():
            raise DataError(f"Training file not found: {p}")
        df = pd.read_csv(p)
        missing = [
            getattr(cfg, col) for col in REQUIRED_COLUMNS if getattr(cfg, col) not in df.columns
        ]
        if missing:
            raise DataError(
                f"{p}: missing required column(s) {missing}; "
                f"available: {list(df.columns)}"
            )
        df["__source_file"] = p.name
        frames.append(df)
    if not frames:
        raise DataError("No training files supplied")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Apply length filters, NA handling, dedup, and label whitelisting."""
    n0 = len(df)
    text = cfg.text_column
    label = cfg.label_column

    # Drop rows with NA text.
    df = df.dropna(subset=[text]).copy()
    df[text] = df[text].astype(str).str.strip()

    if cfg.dropna_labels:
        df = df.dropna(subset=[label])

    # Keep only labels in the configured order.
    allowed = set(cfg.label_order)
    df = df[df[label].isin(allowed)]

    # Length filters.
    lens = df[text].str.len()
    df = df[lens >= cfg.min_text_chars]
    if cfg.max_text_chars is not None:
        df = df[df[text].str.len() <= cfg.max_text_chars]

    if cfg.dedupe_text:
        df = df.drop_duplicates(subset=[text])

    log.info(
        "Cleaning: %d -> %d rows (dropped %d). Label counts: %s",
        n0,
        len(df),
        n0 - len(df),
        Counter(df[label]).most_common(),
    )
    if df.empty:
        raise DataError("All rows were filtered out during cleaning")
    return df.reset_index(drop=True)


def sample_per_class(df: pd.DataFrame, cfg: DataConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Take at most ``sample_per_class`` rows per class.

    Classes with fewer rows than the cap are kept entirely (this matches the
    SetFit paper's few-shot convention).
    """
    if cfg.sample_per_class is None:
        return df

    label = cfg.label_column
    chunks: list[pd.DataFrame] = []
    for lab in cfg.label_order:
        rows = df[df[label] == lab]
        if rows.empty:
            log.warning("No training rows for label %r after cleaning", lab)
            continue
        if len(rows) > cfg.sample_per_class:
            idx = rng.choice(len(rows), size=cfg.sample_per_class, replace=False)
            rows = rows.iloc[idx]
        chunks.append(rows)
    out = pd.concat(chunks, ignore_index=True)
    log.info(
        "Sampled %d rows (<= %d per class). Counts: %s",
        len(out),
        cfg.sample_per_class,
        Counter(out[label]).most_common(),
    )
    return out


def train_eval_split(
    df: pd.DataFrame, cfg: DataConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve out an in-distribution eval slice from the training pool."""
    if cfg.eval_split_ratio == 0.0:
        return df, df.iloc[0:0].copy()

    label = cfg.label_column

    if cfg.stratify_eval:
        eval_parts: list[pd.DataFrame] = []
        train_parts: list[pd.DataFrame] = []
        for lab, group in df.groupby(label, sort=False):
            n_eval = max(1, int(round(len(group) * cfg.eval_split_ratio)))
            perm = rng.permutation(len(group))
            eval_parts.append(group.iloc[perm[:n_eval]])
            train_parts.append(group.iloc[perm[n_eval:]])
        eval_df = pd.concat(eval_parts, ignore_index=True)
        train_df = pd.concat(train_parts, ignore_index=True)
    else:
        perm = rng.permutation(len(df))
        n_eval = int(round(len(df) * cfg.eval_split_ratio))
        eval_df = df.iloc[perm[:n_eval]].reset_index(drop=True)
        train_df = df.iloc[perm[n_eval:]].reset_index(drop=True)

    if cfg.eval_max_size is not None and len(eval_df) > cfg.eval_max_size:
        idx = rng.choice(len(eval_df), size=cfg.eval_max_size, replace=False)
        eval_df = eval_df.iloc[idx].reset_index(drop=True)

    log.info("Train/eval split: %d / %d rows", len(train_df), len(eval_df))
    return train_df, eval_df


def prepare_training_data(
    cfg: DataConfig, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """End-to-end: load → clean → sample_per_class → split → return both frames.

    Returns the train df, the eval df, and a manifest dict summarising what
    was kept (for reproducibility records).
    """
    rng = np.random.default_rng(seed)
    raw = load_and_concat(cfg.train_files, cfg)
    cleaned = clean(raw, cfg)
    sampled = sample_per_class(cleaned, cfg, rng)
    train_df, eval_df = train_eval_split(sampled, cfg, rng)

    manifest = {
        "train_files": [str(Path(f).resolve()) for f in cfg.train_files],
        "rows_raw": int(len(raw)),
        "rows_cleaned": int(len(cleaned)),
        "rows_sampled": int(len(sampled)),
        "rows_train": int(len(train_df)),
        "rows_eval": int(len(eval_df)),
        "label_counts_train": {
            str(k): int(v) for k, v in Counter(train_df[cfg.label_column]).items()
        },
        "label_counts_eval": {
            str(k): int(v) for k, v in Counter(eval_df[cfg.label_column]).items()
        },
    }
    return train_df, eval_df, manifest


def load_test_file(path: str | Path, cfg: DataConfig) -> pd.DataFrame:
    """Load a CSV for inference. Labels are optional; if absent, metrics are skipped."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"Test file not found: {p}")
    df = pd.read_csv(p)
    if cfg.text_column not in df.columns:
        raise DataError(
            f"{p}: missing text column {cfg.text_column!r}; available: {list(df.columns)}"
        )
    df = df.copy()
    df[cfg.text_column] = df[cfg.text_column].astype(str).str.strip()
    df = df[df[cfg.text_column].str.len() >= cfg.min_text_chars]
    return df.reset_index(drop=True)
