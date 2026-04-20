"""SetFit training pipeline orchestrator."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from setfit_cefr.config import Config, canonical_yaml
from setfit_cefr.data import prepare_training_data
from setfit_cefr.hashing import file_fingerprint, model_hash

log = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _frame_to_dataset(df: pd.DataFrame, cfg: Config):
    """DataFrame → datasets.Dataset with 'text' and 'label' columns.

    SetFit's Trainer expects columns literally named ``text`` and ``label``,
    or a ``column_mapping`` passed to it. We rename up-front so the mapping
    stays implicit and downstream code can rely on the canonical names.
    """
    from datasets import Dataset

    sub = df[[cfg.data.text_column, cfg.data.label_column]].rename(
        columns={cfg.data.text_column: "text", cfg.data.label_column: "label"}
    )
    return Dataset.from_pandas(sub, preserve_index=False)


def train(config: Config) -> Path:
    """Run a full training job and return the output folder.

    The output folder is ``<runtime.output_root>/<model-hash>/`` where the hash
    is derived from the resolved config (see :mod:`setfit_cefr.hashing`).
    Re-running with the same config overwrites the same folder.
    """
    from setfit import SetFitModel, Trainer, TrainingArguments

    # 1. Identity & paths.
    m_hash = model_hash(config)
    out_dir = Path(config.runtime.output_root) / m_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Model hash %s -> %s", m_hash, out_dir)

    # 2. Seed & device.
    _seed_everything(config.training.seed)
    device = _resolve_device(config.runtime.device)
    log.info("Device: %s", device)

    # 3. Data prep.
    train_df, eval_df, data_manifest = prepare_training_data(
        config.data, seed=config.training.seed
    )
    train_ds = _frame_to_dataset(train_df, config)
    eval_ds = _frame_to_dataset(eval_df, config) if len(eval_df) else None

    # 4. Persist the resolved config + manifests before training so even a
    # crash mid-run leaves reproducible breadcrumbs.
    (out_dir / "config.yaml").write_text(canonical_yaml(config), encoding="utf-8")
    (out_dir / "config.hash").write_text(m_hash + "\n", encoding="utf-8")
    (out_dir / "labels.json").write_text(
        json.dumps(config.data.label_order, indent=2), encoding="utf-8"
    )
    train_fps = [file_fingerprint(f) for f in config.data.train_files]
    (out_dir / "train_manifest.json").write_text(
        json.dumps({**data_manifest, "file_fingerprints": train_fps}, indent=2),
        encoding="utf-8",
    )

    # 5. Build model.
    model = SetFitModel.from_pretrained(
        config.model.pretrained_model,
        labels=list(config.data.label_order),
        multi_target_strategy=config.model.multi_target_strategy,
        use_differentiable_head=(config.model.head_type == "setfit_head"),
        cache_dir=config.runtime.cache_dir,
        device=device,
    )

    # 6. Build trainer. ``checkpoint_dir`` is sent to a nested path so the
    # top-level model folder stays clean.
    effective_eval_strategy = config.training.eval_strategy if eval_ds is not None else "no"
    effective_save_strategy = config.training.save_strategy if eval_ds is not None else "no"
    effective_load_best = config.training.load_best_model_at_end and eval_ds is not None

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        num_iterations=config.training.num_iterations,
        body_learning_rate=config.training.body_learning_rate,
        head_learning_rate=config.training.head_learning_rate,
        warmup_proportion=config.training.warmup_proportion,
        l2_weight=config.training.l2_weight,
        eval_strategy=effective_eval_strategy,
        save_strategy=effective_save_strategy,
        load_best_model_at_end=effective_load_best,
        use_amp=config.training.use_amp,
        max_length=config.training.max_length,
        seed=config.training.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        metric=config.training.metric,
    )

    # 7. Train.
    trainer.train()

    # 8. Final eval if we have a held-out set.
    eval_metrics: dict[str, float] = {}
    if eval_ds is not None:
        eval_metrics = trainer.evaluate(eval_ds)
        log.info("Eval metrics: %s", eval_metrics)
    (out_dir / "eval_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2), encoding="utf-8"
    )

    # 9. Save the trained model in a predictable subfolder.
    model_dir = out_dir / "setfit-model"
    model.save_pretrained(str(model_dir))

    # 10. Persist the trainer state log (epoch-by-epoch metrics).
    try:
        log_history = trainer.state.log_history  # type: ignore[attr-defined]
    except AttributeError:
        log_history = []
    (out_dir / "training_log.json").write_text(
        json.dumps(log_history, indent=2, default=str), encoding="utf-8"
    )

    log.info("Training complete. Artifacts in %s", out_dir)
    return out_dir
