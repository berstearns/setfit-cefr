"""Composable, strictly-validated configuration for setfit-cefr.

Each subsection of the run is a frozen-ish dataclass with an aggressive
``__post_init__`` that rejects bad values *before* any model is loaded. The
top-level :class:`Config` then performs cross-section consistency checks.

Design goals:

* Subconfigs are independently meaningful and testable.
* All validation is eager: constructing a Config either succeeds with a sane
  object or raises :class:`ConfigValidationError` with a precise, dot-pathed
  message. Downstream code never has to re-check the same invariants.
* Serialization is deterministic (``to_dict`` preserves field order) so the
  model-hash is reproducible.
* YAML loading, CLI flag overrides, and arbitrary ``key.path=value`` overrides
  all go through the same ``Config.from_sources`` entry point.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConfigValidationError(ValueError):
    """Raised when a Config or subconfig fails validation.

    The message is always prefixed with the dot-path of the offending field to
    make debugging YAML/override combinations easy.
    """


def _fail(path: str, msg: str) -> None:
    raise ConfigValidationError(f"{path}: {msg}")


# ---------------------------------------------------------------------------
# Allowed enums (kept in one place so the validators and error messages agree)
# ---------------------------------------------------------------------------

HEAD_TYPES = ("logistic", "setfit_head")
MULTI_TARGET_STRATEGIES = (None, "one-vs-rest", "multi-output", "classifier-chain")
EVAL_STRATEGIES = ("no", "steps", "epoch")
SAVE_STRATEGIES = ("no", "steps", "epoch")
METRICS = ("accuracy", "f1", "matthews_correlation")
DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:\d+)?)$")
# Matches common ISO-2 / ISO-3 language identifiers plus a handful of CEFR
# corpus conventions; kept forgiving because real data has long tails.
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._\-+@:/]+$")


# ---------------------------------------------------------------------------
# Module-level toggles
# ---------------------------------------------------------------------------

# The only global state in this module. Set to False during prediction so
# Config.from_sources can load a training config that references absolute paths
# no longer valid on this machine. Kept out of the dataclass state so it does
# not contaminate the canonical YAML / model hash.
_FILE_EXISTENCE_CHECKS = True


def set_file_existence_checks(enabled: bool) -> None:
    """Toggle the ``DataConfig.train_files`` existence checks.

    Only flip this when you intentionally want to build a Config whose training
    file paths are stale (e.g. when running prediction from a different machine
    than where the model was trained).
    """
    global _FILE_EXISTENCE_CHECKS
    _FILE_EXISTENCE_CHECKS = bool(enabled)


# ---------------------------------------------------------------------------
# Subconfigs
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Where training data lives and how rows are filtered / sampled."""

    train_files: list[str] = field(default_factory=list)
    text_column: str = "text"
    label_column: str = "cefr_level"
    id_column: str = "writing_id"
    label_order: list[str] = field(
        default_factory=lambda: ["A1", "A2", "B1", "B2", "C1", "C2"]
    )
    min_text_chars: int = 1
    max_text_chars: int | None = None
    dropna_labels: bool = True
    dedupe_text: bool = True
    sample_per_class: int | None = None
    eval_split_ratio: float = 0.05
    eval_max_size: int | None = 1500
    stratify_eval: bool = True

    def __post_init__(self) -> None:
        p = "data"

        if not isinstance(self.train_files, list) or not self.train_files:
            _fail(f"{p}.train_files", "must be a non-empty list of paths")
        seen: set[str] = set()
        for i, f in enumerate(self.train_files):
            sub = f"{p}.train_files[{i}]"
            if not isinstance(f, str) or not f.strip():
                _fail(sub, "must be a non-empty string")
            if f in seen:
                _fail(sub, f"duplicate path {f!r}")
            seen.add(f)
            if _FILE_EXISTENCE_CHECKS and not Path(f).exists():
                _fail(sub, f"file does not exist: {f}")

        for name in ("text_column", "label_column", "id_column"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                _fail(f"{p}.{name}", "must be a non-empty string")
        if len({self.text_column, self.label_column, self.id_column}) != 3:
            _fail(
                p,
                "text_column, label_column and id_column must all differ "
                f"(got text={self.text_column!r}, label={self.label_column!r}, "
                f"id={self.id_column!r})",
            )

        if not isinstance(self.label_order, list) or not self.label_order:
            _fail(f"{p}.label_order", "must be a non-empty list")
        if len(set(self.label_order)) != len(self.label_order):
            _fail(f"{p}.label_order", "must contain unique entries")
        for i, lab in enumerate(self.label_order):
            if not isinstance(lab, str) or not lab.strip():
                _fail(f"{p}.label_order[{i}]", "must be a non-empty string")

        if not isinstance(self.min_text_chars, int) or self.min_text_chars < 1:
            _fail(f"{p}.min_text_chars", "must be an int >= 1")
        if self.max_text_chars is not None:
            if not isinstance(self.max_text_chars, int) or self.max_text_chars < 1:
                _fail(f"{p}.max_text_chars", "must be null or an int >= 1")
            if self.max_text_chars < self.min_text_chars:
                _fail(
                    f"{p}.max_text_chars",
                    f"({self.max_text_chars}) must be >= min_text_chars ({self.min_text_chars})",
                )

        if not isinstance(self.dropna_labels, bool):
            _fail(f"{p}.dropna_labels", "must be bool")
        if not isinstance(self.dedupe_text, bool):
            _fail(f"{p}.dedupe_text", "must be bool")

        if self.sample_per_class is not None and (
            not isinstance(self.sample_per_class, int) or self.sample_per_class < 1
        ):
            _fail(f"{p}.sample_per_class", "must be null or an int >= 1")

        if not isinstance(self.eval_split_ratio, (int, float)):
            _fail(f"{p}.eval_split_ratio", "must be a float")
        if not (0.0 <= float(self.eval_split_ratio) < 0.5):
            _fail(f"{p}.eval_split_ratio", "must be in [0.0, 0.5)")

        if self.eval_max_size is not None and (
            not isinstance(self.eval_max_size, int) or self.eval_max_size < 1
        ):
            _fail(f"{p}.eval_max_size", "must be null or an int >= 1")

        if not isinstance(self.stratify_eval, bool):
            _fail(f"{p}.stratify_eval", "must be bool")


@dataclass
class ModelConfig:
    """SetFit backbone + head configuration."""

    pretrained_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    head_type: str = "logistic"
    multi_target_strategy: str | None = None

    def __post_init__(self) -> None:
        p = "model"

        if not isinstance(self.pretrained_model, str) or not self.pretrained_model.strip():
            _fail(f"{p}.pretrained_model", "must be a non-empty string")
        if not SAFE_NAME_RE.match(self.pretrained_model):
            _fail(
                f"{p}.pretrained_model",
                f"{self.pretrained_model!r} contains unsafe characters",
            )

        if self.head_type not in HEAD_TYPES:
            _fail(f"{p}.head_type", f"must be one of {HEAD_TYPES}, got {self.head_type!r}")

        if self.multi_target_strategy not in MULTI_TARGET_STRATEGIES:
            _fail(
                f"{p}.multi_target_strategy",
                f"must be one of {MULTI_TARGET_STRATEGIES}, got {self.multi_target_strategy!r}",
            )


@dataclass
class TrainingConfig:
    """Optimisation & SetFit trainer knobs."""

    num_epochs: int = 1
    batch_size: int = 16
    body_learning_rate: float = 2e-5
    head_learning_rate: float = 1e-2
    num_iterations: int = 20
    seed: int = 42
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric: str = "accuracy"
    warmup_proportion: float = 0.1
    l2_weight: float = 0.01
    use_amp: bool = False
    max_length: int | None = 256

    def __post_init__(self) -> None:
        p = "training"

        for name in ("num_epochs", "batch_size", "num_iterations"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                _fail(f"{p}.{name}", f"must be an int >= 1, got {v!r}")

        for name in ("body_learning_rate", "head_learning_rate"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not (0 < float(v) < 1):
                _fail(f"{p}.{name}", f"must be a float in (0, 1), got {v!r}")

        if not isinstance(self.seed, int) or self.seed < 0:
            _fail(f"{p}.seed", "must be an int >= 0")

        if self.eval_strategy not in EVAL_STRATEGIES:
            _fail(f"{p}.eval_strategy", f"must be one of {EVAL_STRATEGIES}")
        if self.save_strategy not in SAVE_STRATEGIES:
            _fail(f"{p}.save_strategy", f"must be one of {SAVE_STRATEGIES}")

        if not isinstance(self.load_best_model_at_end, bool):
            _fail(f"{p}.load_best_model_at_end", "must be bool")
        if self.load_best_model_at_end:
            if self.eval_strategy == "no":
                _fail(
                    f"{p}.load_best_model_at_end",
                    "cannot be true when eval_strategy == 'no' (no checkpoints to pick from)",
                )
            if self.save_strategy != self.eval_strategy:
                _fail(
                    f"{p}.load_best_model_at_end",
                    f"requires save_strategy == eval_strategy, got "
                    f"save={self.save_strategy!r} eval={self.eval_strategy!r}",
                )

        if self.metric not in METRICS:
            _fail(f"{p}.metric", f"must be one of {METRICS}, got {self.metric!r}")

        if not isinstance(self.warmup_proportion, (int, float)) or not (
            0.0 <= float(self.warmup_proportion) <= 1.0
        ):
            _fail(f"{p}.warmup_proportion", "must be a float in [0, 1]")

        if not isinstance(self.l2_weight, (int, float)) or float(self.l2_weight) < 0:
            _fail(f"{p}.l2_weight", "must be a non-negative float")

        if not isinstance(self.use_amp, bool):
            _fail(f"{p}.use_amp", "must be bool")

        if self.max_length is not None and (
            not isinstance(self.max_length, int) or self.max_length < 1
        ):
            _fail(f"{p}.max_length", "must be null or an int >= 1")


@dataclass
class RuntimeConfig:
    """Where artefacts go and how the hardware is selected."""

    device: str = "auto"
    output_root: str = "models"
    predictions_root: str = "predictions"
    cache_dir: str | None = None
    num_workers: int = 0

    def __post_init__(self) -> None:
        p = "runtime"

        if not isinstance(self.device, str) or not DEVICE_RE.match(self.device):
            _fail(
                f"{p}.device",
                f"must match 'auto' | 'cpu' | 'cuda' | 'cuda:N', got {self.device!r}",
            )

        for name in ("output_root", "predictions_root"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                _fail(f"{p}.{name}", "must be a non-empty string")

        if self.cache_dir is not None and (
            not isinstance(self.cache_dir, str) or not self.cache_dir.strip()
        ):
            _fail(f"{p}.cache_dir", "must be null or a non-empty string")

        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            _fail(f"{p}.num_workers", "must be an int >= 0")


@dataclass
class ReportingConfig:
    """What metrics and artefacts to compute in the evaluation report."""

    compute_macro_f1: bool = True
    compute_qwk: bool = True
    compute_adjacent_accuracy: bool = True
    save_confusion_matrix: bool = True
    top_k_examples: int = 5

    def __post_init__(self) -> None:
        p = "reporting"
        for name in (
            "compute_macro_f1",
            "compute_qwk",
            "compute_adjacent_accuracy",
            "save_confusion_matrix",
        ):
            if not isinstance(getattr(self, name), bool):
                _fail(f"{p}.{name}", "must be bool")
        if not isinstance(self.top_k_examples, int) or self.top_k_examples < 0:
            _fail(f"{p}.top_k_examples", "must be an int >= 0")


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    run_name: str | None = None
    experiment_tag: str = "default"

    def __post_init__(self) -> None:
        # If subconfigs were passed in as dicts (YAML path), convert them.
        self.data = _coerce(self.data, DataConfig, "data")
        self.model = _coerce(self.model, ModelConfig, "model")
        self.training = _coerce(self.training, TrainingConfig, "training")
        self.runtime = _coerce(self.runtime, RuntimeConfig, "runtime")
        self.reporting = _coerce(self.reporting, ReportingConfig, "reporting")

        if self.run_name is not None:
            if not isinstance(self.run_name, str) or not self.run_name.strip():
                _fail("run_name", "must be null or a non-empty string")
            if not SAFE_NAME_RE.match(self.run_name):
                _fail("run_name", f"{self.run_name!r} contains unsafe characters")
        if not isinstance(self.experiment_tag, str) or not self.experiment_tag.strip():
            _fail("experiment_tag", "must be a non-empty string")
        if not SAFE_NAME_RE.match(self.experiment_tag):
            _fail("experiment_tag", f"{self.experiment_tag!r} contains unsafe characters")

        # Cross-section invariants. These are the checks that no subconfig
        # could do in isolation.
        if self.data.eval_split_ratio == 0.0 and self.training.load_best_model_at_end:
            _fail(
                "training.load_best_model_at_end",
                "cannot be true when data.eval_split_ratio == 0.0 (no eval set)",
            )
        if self.data.eval_split_ratio == 0.0 and self.training.eval_strategy != "no":
            _fail(
                "training.eval_strategy",
                f"must be 'no' when data.eval_split_ratio == 0.0, got {self.training.eval_strategy!r}",
            )
        if (
            self.data.sample_per_class is not None
            and self.training.batch_size > self.data.sample_per_class * len(self.data.label_order)
        ):
            _fail(
                "training.batch_size",
                f"({self.training.batch_size}) exceeds total sampled training size "
                f"({self.data.sample_per_class} * {len(self.data.label_order)} classes)",
            )

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    # -- factories ----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        return cls(**data)  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ConfigValidationError(f"{path}: top-level YAML must be a mapping")
        return cls.from_dict(raw)

    @classmethod
    def from_sources(
        cls,
        yaml_path: str | os.PathLike[str] | None = None,
        flag_overrides: Mapping[str, Any] | None = None,
        dotted_overrides: list[str] | None = None,
    ) -> "Config":
        """Build a Config by layering: YAML → flag overrides → dotted overrides.

        ``flag_overrides`` is a dict of already-parsed CLI flags (typed
        Python values). ``dotted_overrides`` is a list of strings of the form
        ``"a.b.c=value"`` where ``value`` is parsed as YAML so ints, bools and
        lists all work transparently.
        """
        base: dict[str, Any] = {}
        if yaml_path is not None:
            with open(yaml_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ConfigValidationError(f"{yaml_path}: top-level YAML must be a mapping")
            base = loaded

        if flag_overrides:
            for dotted, value in flag_overrides.items():
                if value is None:
                    continue
                _assign_dotted(base, dotted, value)

        if dotted_overrides:
            for entry in dotted_overrides:
                if "=" not in entry:
                    raise ConfigValidationError(
                        f"--override entries must be of the form KEY.PATH=VALUE, got {entry!r}"
                    )
                key, _, raw = entry.partition("=")
                try:
                    value = yaml.safe_load(raw)
                except yaml.YAMLError as e:
                    raise ConfigValidationError(
                        f"--override {entry!r}: could not parse value as YAML: {e}"
                    ) from e
                _assign_dotted(base, key.strip(), value)

        return cls.from_dict(base)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce(value: Any, target_cls: type, path: str) -> Any:
    """Coerce a dict (from YAML) into its target dataclass, tagging the dot-path on error."""
    if isinstance(value, target_cls):
        return value
    if isinstance(value, Mapping):
        # Reject unknown keys eagerly: typos in YAML are the #1 source of
        # surprises and silently-ignored overrides.
        known = {f.name for f in fields(target_cls) if not f.name.startswith("_")}
        unknown = set(value.keys()) - known
        if unknown:
            _fail(path, f"unknown field(s): {sorted(unknown)} (allowed: {sorted(known)})")
        return target_cls(**value)
    _fail(path, f"must be a mapping for {target_cls.__name__}, got {type(value).__name__}")


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursive, field-order-preserving dataclass -> plain dict.

    Unlike :func:`dataclasses.asdict` this skips private (underscore-prefixed)
    fields so they don't leak into the hashed canonical form.
    """
    if is_dataclass(obj):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            if f.name.startswith("_"):
                continue
            out[f.name] = _dataclass_to_dict(getattr(obj, f.name))
        return out
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, Mapping):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def _assign_dotted(root: dict[str, Any], dotted_key: str, value: Any) -> None:
    """``root["a"]["b"]["c"] = value`` given ``dotted_key == "a.b.c"``."""
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ConfigValidationError("override key must not be empty")
    cur: Any = root
    for part in parts[:-1]:
        nxt = cur.get(part)
        if nxt is None or not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = copy.deepcopy(value)


def canonical_yaml(config: Config) -> str:
    """Deterministic YAML representation used for hashing.

    Lists and dicts keep their field order (determined by the dataclass
    definitions), which gives stable hashes across runs on the same config.
    """
    return yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=False)
