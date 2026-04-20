"""Validation & layering tests for ``NotebookConfig``."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from setfit_cefr import NotebookConfig
from setfit_cefr.config import ConfigValidationError
from setfit_cefr.notebook_config import InputConfig, OutputConfig, RunConfig


# ---------------------------------------------------------------------------
# InputConfig
# ---------------------------------------------------------------------------


def test_input_defaults_are_valid():
    cfg = InputConfig()
    assert set(cfg.train_keys) <= set(cfg.filenames)
    assert set(cfg.test_keys) <= set(cfg.filenames)


def test_input_rejects_empty_data_dir():
    with pytest.raises(ConfigValidationError, match="data_dir"):
        InputConfig(data_dir="")


def test_input_rejects_empty_filenames():
    with pytest.raises(ConfigValidationError, match="filenames"):
        InputConfig(filenames={})


def test_input_rejects_key_not_in_filenames():
    with pytest.raises(ConfigValidationError, match="unknown filenames key"):
        InputConfig(
            filenames={"a": "a.csv"},
            train_keys=["a", "b"],
            test_keys=["a"],
        )


def test_input_rejects_overlap_between_train_and_test():
    with pytest.raises(ConfigValidationError, match="overlap"):
        InputConfig(
            filenames={"x": "x.csv", "y": "y.csv"},
            train_keys=["x"],
            test_keys=["x", "y"],
        )


def test_input_rejects_duplicate_keys_in_list():
    with pytest.raises(ConfigValidationError, match="unique"):
        InputConfig(
            filenames={"a": "a.csv", "b": "b.csv"},
            train_keys=["a", "a"],
            test_keys=["b"],
        )


# ---------------------------------------------------------------------------
# OutputConfig
# ---------------------------------------------------------------------------


def test_output_rejects_empty_repo_url():
    with pytest.raises(ConfigValidationError, match="repo_url"):
        OutputConfig(repo_url="")


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


def test_run_allows_null_config_path():
    RunConfig(config_path=None)


def test_run_rejects_bad_override_string():
    with pytest.raises(ConfigValidationError, match="dotted_overrides"):
        RunConfig(dotted_overrides=["no_equals_sign"])


def test_run_rejects_non_string_flag():
    with pytest.raises(ConfigValidationError, match="flag_args"):
        RunConfig(flag_args=["--epochs", 3])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Top-level + factories
# ---------------------------------------------------------------------------


def test_notebook_config_defaults():
    cfg = NotebookConfig()
    assert cfg.input.data_dir
    assert cfg.output.repo_url.startswith("https://")
    assert cfg.train.config_path == "configs/default.yaml"
    assert cfg.predict.config_path is None


def test_notebook_config_from_sources_layers_overrides(tmp_path: Path):
    y = tmp_path / "nb.yaml"
    y.write_text(
        textwrap.dedent(
            """
            input:
              data_dir: /tmp
            train:
              flag_args: []
            """
        ).strip()
    )
    cfg = NotebookConfig.from_sources(
        yaml_path=y,
        overrides={
            "input.data_dir": "/data",
            "train.flag_args": ["--epochs", "3"],
            "train.dotted_overrides": ["training.max_length=256"],
            "predict.config_path": None,
        },
    )
    assert cfg.input.data_dir == "/data"
    assert cfg.train.flag_args == ["--epochs", "3"]
    assert cfg.train.dotted_overrides == ["training.max_length=256"]
    assert cfg.predict.config_path is None


def test_notebook_config_derived_paths():
    cfg = NotebookConfig()
    train_paths = cfg.local_train_files()
    test_paths = cfg.local_test_files()
    assert all(str(p).startswith(cfg.output.local_data_dir) for p in train_paths)
    assert all(str(p).startswith(cfg.output.local_data_dir) for p in test_paths)
    # No file is claimed by both splits.
    assert not (set(train_paths) & set(test_paths))


def test_notebook_config_to_yaml_roundtrip(tmp_path: Path):
    cfg = NotebookConfig.from_sources(
        overrides={
            "input.data_dir": "/data",
            "train.flag_args": ["--seed", "7"],
        }
    )
    out = tmp_path / "roundtrip.yaml"
    cfg.write_yaml(out)
    cfg2 = NotebookConfig.from_yaml(out)
    assert cfg2.input.data_dir == "/data"
    assert cfg2.train.flag_args == ["--seed", "7"]


def test_notebook_config_rejects_unknown_top_level_key():
    with pytest.raises(ConfigValidationError, match="unknown field"):
        NotebookConfig(input={"not_a_field": "x"})  # type: ignore[arg-type]


def test_shipped_notebook_yaml_is_valid():
    """The committed configs/notebook.yaml must validate out of the box."""
    repo_root = Path(__file__).resolve().parent.parent
    cfg = NotebookConfig.from_yaml(repo_root / "configs" / "notebook.yaml")
    assert cfg.input.filenames["train"].endswith(".csv")
    assert "test" in cfg.input.test_keys
