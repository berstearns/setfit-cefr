"""Notebook-level configuration.

Governs the end-to-end flow of ``train.ipynb`` (and any future notebook in this
project). Separate from :class:`setfit_cefr.config.Config` — that class captures
*what a model is* (backbone, hyperparams, data columns) while this one captures
*how the notebook should behave* (where CSVs live, where to write outputs,
which CLI flags to forward to ``train.py`` / ``predict.py``).

Design rules (same as ``config.py``):

* Composable subconfigs (:class:`InputConfig`, :class:`OutputConfig`,
  :class:`RunConfig`).
* ``__post_init__`` validation with dot-pathed :class:`ConfigValidationError`.
* Single ``from_sources`` factory layers YAML defaults → dotted overrides.
* No secret state: everything serialises deterministically via ``to_yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from setfit_cefr.config import (
    ConfigValidationError,
    _assign_dotted,
    _coerce,
    _dataclass_to_dict,
    _fail,
)


# ---------------------------------------------------------------------------
# Subconfigs
# ---------------------------------------------------------------------------


@dataclass
class InputConfig:
    """Where the CEFR CSVs live and which roles they play."""

    data_dir: str = "/path/to/your/csvs"
    filenames: dict[str, str] = field(
        default_factory=lambda: {
            "train": "norm-EFCAMDAT-train.csv",
            "remainder": "norm-EFCAMDAT-remainder.csv",
            "test": "norm-EFCAMDAT-test.csv",
            "kupa": "norm-KUPA-KEYS.csv",
            "celva": "norm-CELVA-SP.csv",
        }
    )
    train_keys: list[str] = field(default_factory=lambda: ["train", "remainder"])
    test_keys: list[str] = field(default_factory=lambda: ["test", "kupa", "celva"])

    def __post_init__(self) -> None:
        p = "input"
        if not isinstance(self.data_dir, str) or not self.data_dir.strip():
            _fail(f"{p}.data_dir", "must be a non-empty string")
        if not isinstance(self.filenames, dict) or not self.filenames:
            _fail(f"{p}.filenames", "must be a non-empty mapping")
        for k, v in self.filenames.items():
            if not isinstance(k, str) or not k.strip():
                _fail(f"{p}.filenames", f"key {k!r} must be a non-empty string")
            if not isinstance(v, str) or not v.strip():
                _fail(f"{p}.filenames[{k}]", "must be a non-empty string")
        for list_name in ("train_keys", "test_keys"):
            lst = getattr(self, list_name)
            if not isinstance(lst, list) or not lst:
                _fail(f"{p}.{list_name}", "must be a non-empty list")
            if len(set(lst)) != len(lst):
                _fail(f"{p}.{list_name}", "must contain unique entries")
            missing = [k for k in lst if k not in self.filenames]
            if missing:
                _fail(
                    f"{p}.{list_name}",
                    f"references unknown filenames key(s): {missing}; "
                    f"known: {sorted(self.filenames.keys())}",
                )
        overlap = set(self.train_keys) & set(self.test_keys)
        if overlap:
            _fail(
                p,
                f"train_keys and test_keys overlap on {sorted(overlap)}; a file cannot be both",
            )


@dataclass
class OutputConfig:
    """Repo, clone target, and where models/predictions land."""

    repo_url: str = "https://github.com/berstearns/setfit-cefr.git"
    repo_dir: str = "setfit-cefr"
    local_data_dir: str = "data"
    models_root: str = "models"
    predictions_root: str = "predictions"

    def __post_init__(self) -> None:
        p = "output"
        for name in ("repo_url", "repo_dir", "local_data_dir", "models_root", "predictions_root"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                _fail(f"{p}.{name}", "must be a non-empty string")


@dataclass
class RunConfig:
    """What to pass to ``train.py`` / ``predict.py``."""

    # Optional YAML config file for train.py / predict.py. Predict defaults to
    # None, which makes predict.py fall back to the trained model's saved
    # config.yaml.
    config_path: str | None = "configs/default.yaml"
    # Flat CLI flags (e.g. ["--epochs", "3", "--seed", "7"]).
    flag_args: list[str] = field(default_factory=list)
    # Dotted overrides in the `key.path=value` form used by --override.
    dotted_overrides: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        p = "run"
        if self.config_path is not None and (
            not isinstance(self.config_path, str) or not self.config_path.strip()
        ):
            _fail(f"{p}.config_path", "must be null or a non-empty string")
        if not isinstance(self.flag_args, list):
            _fail(f"{p}.flag_args", "must be a list of strings")
        for i, a in enumerate(self.flag_args):
            if not isinstance(a, str):
                _fail(f"{p}.flag_args[{i}]", "must be a string")
        if not isinstance(self.dotted_overrides, list):
            _fail(f"{p}.dotted_overrides", "must be a list of strings")
        for i, a in enumerate(self.dotted_overrides):
            if not isinstance(a, str) or "=" not in a:
                _fail(
                    f"{p}.dotted_overrides[{i}]",
                    "must be a 'key.path=value' string",
                )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@dataclass
class NotebookConfig:
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    train: RunConfig = field(default_factory=RunConfig)
    predict: RunConfig = field(
        default_factory=lambda: RunConfig(config_path=None)
    )

    def __post_init__(self) -> None:
        self.input = _coerce(self.input, InputConfig, "input")
        self.output = _coerce(self.output, OutputConfig, "output")
        self.train = _coerce(self.train, RunConfig, "train")
        self.predict = _coerce(self.predict, RunConfig, "predict")

    # -- derived paths ------------------------------------------------------

    def resolved_data_dir(self) -> Path:
        return Path(self.input.data_dir).expanduser().resolve()

    def local_train_files(self) -> list[Path]:
        """Paths inside ``output.local_data_dir`` for training CSVs."""
        base = Path(self.output.local_data_dir)
        return [base / self.input.filenames[k] for k in self.input.train_keys]

    def local_test_files(self) -> list[Path]:
        """Paths inside ``output.local_data_dir`` for external test CSVs."""
        base = Path(self.output.local_data_dir)
        return [base / self.input.filenames[k] for k in self.input.test_keys]

    def source_file_pairs(self) -> list[tuple[Path, Path]]:
        """``(src_under_data_dir, dst_under_local_data_dir)`` for all files, train+test.

        Used by the notebook's symlink/copy step.
        """
        src = self.resolved_data_dir()
        dst = Path(self.output.local_data_dir)
        keys = list(self.input.train_keys) + list(self.input.test_keys)
        return [(src / self.input.filenames[k], dst / self.input.filenames[k]) for k in keys]

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def write_yaml(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    # -- factories ----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NotebookConfig":
        return cls(**data)  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NotebookConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ConfigValidationError(f"{path}: top-level YAML must be a mapping")
        return cls.from_dict(raw)

    @classmethod
    def from_sources(
        cls,
        yaml_path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "NotebookConfig":
        """YAML defaults + dotted-path overrides. The one knob the notebook uses.

        ``overrides`` is a mapping whose keys are dotted paths (e.g.
        ``"input.data_dir"``, ``"train.flag_args"``) and whose values are the
        target Python values (strings, lists, etc.). This keeps the notebook's
        single user-facing cell compact.
        """
        base: dict[str, Any] = {}
        if yaml_path is not None:
            with open(yaml_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ConfigValidationError(f"{yaml_path}: top-level YAML must be a mapping")
            base = loaded
        if overrides:
            for dotted, value in overrides.items():
                if not isinstance(dotted, str) or not dotted.strip():
                    raise ConfigValidationError(
                        f"override key must be a non-empty string, got {dotted!r}"
                    )
                _assign_dotted(base, dotted, value)
        return cls.from_dict(base)
