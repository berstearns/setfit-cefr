"""Shared argparse builders for train.py and predict.py.

Both CLIs follow the same pattern:

  * ``--config PATH``         — YAML config (required for train, optional for predict).
  * flag shortcuts            — e.g. ``--epochs 3``, ``--batch-size 32``.
  * ``--override KEY=VALUE``  — arbitrary dotted-path overrides, repeatable.
  * ``--dry-run``             — resolve the config and exit (useful for hash preview).

Keeping the builder here means both entry points stay tiny and honest: they
really are thin wrappers.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Shared flag mapping
# ---------------------------------------------------------------------------

# flag name -> dotted config path. Only flags declared here are accepted; this
# keeps the CLI surface small and forces ``--override`` for edge cases.
_FLAG_TO_DOTTED: dict[str, str] = {
    "epochs": "training.num_epochs",
    "batch_size": "training.batch_size",
    "num_iterations": "training.num_iterations",
    "body_lr": "training.body_learning_rate",
    "head_lr": "training.head_learning_rate",
    "seed": "training.seed",
    "max_length": "training.max_length",
    "model_name": "model.pretrained_model",
    "head_type": "model.head_type",
    "sample_per_class": "data.sample_per_class",
    "eval_split_ratio": "data.eval_split_ratio",
    "run_name": "run_name",
    "experiment_tag": "experiment_tag",
    "device": "runtime.device",
    "output_root": "runtime.output_root",
    "predictions_root": "runtime.predictions_root",
}


def _add_common_overrides(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("config overrides")
    g.add_argument("--epochs", type=int, dest="epochs", default=None)
    g.add_argument("--batch-size", type=int, dest="batch_size", default=None)
    g.add_argument("--num-iterations", type=int, dest="num_iterations", default=None)
    g.add_argument("--body-lr", type=float, dest="body_lr", default=None)
    g.add_argument("--head-lr", type=float, dest="head_lr", default=None)
    g.add_argument("--seed", type=int, dest="seed", default=None)
    g.add_argument("--max-length", type=int, dest="max_length", default=None)
    g.add_argument("--model-name", type=str, dest="model_name", default=None)
    g.add_argument(
        "--head-type", type=str, choices=("logistic", "setfit_head"), dest="head_type", default=None
    )
    g.add_argument("--sample-per-class", type=int, dest="sample_per_class", default=None)
    g.add_argument("--eval-split-ratio", type=float, dest="eval_split_ratio", default=None)
    g.add_argument("--run-name", type=str, dest="run_name", default=None)
    g.add_argument("--experiment-tag", type=str, dest="experiment_tag", default=None)
    g.add_argument("--device", type=str, dest="device", default=None)
    g.add_argument("--output-root", type=str, dest="output_root", default=None)
    g.add_argument("--predictions-root", type=str, dest="predictions_root", default=None)
    g.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Arbitrary dotted-path override. Repeat for multiple. "
        "Example: --override training.num_epochs=3 --override data.sample_per_class=32",
    )


def _extract_flag_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Pick up only the flags that were explicitly set (not None)."""
    out: dict[str, Any] = {}
    for flag, dotted in _FLAG_TO_DOTTED.items():
        val = getattr(args, flag, None)
        if val is not None:
            out[dotted] = val
    return out


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train a SetFit-CEFR classifier. Output lands in "
        "<runtime.output_root>/<model-hash>/.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config. See configs/default.yaml for the full surface.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve + validate the config and print the model hash, then exit.",
    )
    _add_common_overrides(parser)
    return parser


def train_main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``setfit-cefr-train``."""
    parser = build_train_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    # Local imports so ``--help`` doesn't pay the SetFit/torch import tax.
    from setfit_cefr.config import Config
    from setfit_cefr.hashing import model_hash

    flag_overrides = _extract_flag_overrides(args)
    config = Config.from_sources(
        yaml_path=args.config,
        flag_overrides=flag_overrides,
        dotted_overrides=args.override,
    )

    m_hash = model_hash(config)
    logging.getLogger(__name__).info("Resolved config. Model hash: %s", m_hash)
    if args.dry_run:
        print(m_hash)
        return 0

    from setfit_cefr.training import train as run_train

    out_dir = run_train(config)
    print(str(out_dir))
    return 0


# ---------------------------------------------------------------------------
# predict.py
# ---------------------------------------------------------------------------


def build_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Score one or more test CSVs with a trained SetFit-CEFR model. "
        "Output lands in <runtime.predictions_root>/<predict-hash>/.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a trained model folder (e.g. models/<hash>/).",
    )
    parser.add_argument(
        "--test-files",
        required=True,
        nargs="+",
        help="One or more CSV files to score. Columns must include text_column; "
        "label_column is optional (metrics are skipped when absent).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config. Defaults to the model's saved config.yaml.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve + validate the config and print the predict hash, then exit.",
    )
    _add_common_overrides(parser)
    return parser


def predict_main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``setfit-cefr-predict``."""
    parser = build_predict_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    from pathlib import Path

    from setfit_cefr.config import Config, set_file_existence_checks
    from setfit_cefr.hashing import predict_hash

    model_path = Path(args.model).resolve()
    config_path = args.config
    if config_path is None:
        default_cfg = model_path / "config.yaml"
        if default_cfg.exists():
            config_path = str(default_cfg)

    flag_overrides = _extract_flag_overrides(args)
    # Skip DataConfig.train_files existence checks during predict: the model's
    # saved config.yaml refers to the training CSVs as they were on the
    # training machine, which may not exist on this one.
    set_file_existence_checks(False)
    try:
        if config_path is not None:
            config_source = Config.from_sources(
                yaml_path=config_path,
                flag_overrides=flag_overrides,
                dotted_overrides=list(args.override),
            )
        else:
            # Defaults + overrides; but a DataConfig with default empty
            # train_files would fail validation, so synthesise a placeholder.
            config_source = Config.from_sources(
                flag_overrides=flag_overrides,
                dotted_overrides=list(args.override)
                + [f"data.train_files={args.test_files!r}"],
            )
    finally:
        set_file_existence_checks(True)

    overrides_yaml = "\n".join(sorted(args.override))
    p_hash = predict_hash(
        model_path.name, args.test_files, overrides_yaml
    )
    logging.getLogger(__name__).info("Resolved config. Predict hash: %s", p_hash)
    if args.dry_run:
        print(p_hash)
        return 0

    from setfit_cefr.inference import predict as run_predict

    out_dir = run_predict(
        model_dir=model_path,
        test_files=args.test_files,
        config=config_source,
        predict_overrides_yaml=overrides_yaml,
    )
    print(str(out_dir))
    return 0
