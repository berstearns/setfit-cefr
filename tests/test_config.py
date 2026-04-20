"""Covers the ``Config`` validators and the layered from-sources loader.

These tests exercise only the config module, so they run without SetFit or
torch installed — making them useful as a pre-commit quality gate.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from setfit_cefr.config import (
    Config,
    ConfigValidationError,
    DataConfig,
    ModelConfig,
    ReportingConfig,
    RuntimeConfig,
    TrainingConfig,
    canonical_yaml,
    set_file_existence_checks,
)
from setfit_cefr.hashing import model_hash


@pytest.fixture(autouse=True)
def _disable_file_checks():
    """Tests construct configs with fake paths."""
    set_file_existence_checks(False)
    yield
    set_file_existence_checks(True)


def _good_data(**overrides) -> DataConfig:
    kwargs = dict(train_files=["fake/a.csv"])
    kwargs.update(overrides)
    return DataConfig(**kwargs)


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


def test_data_requires_at_least_one_file():
    with pytest.raises(ConfigValidationError, match="train_files"):
        DataConfig(train_files=[])


def test_data_rejects_duplicate_files():
    with pytest.raises(ConfigValidationError, match="duplicate"):
        DataConfig(train_files=["a.csv", "a.csv"])


def test_data_rejects_identical_columns():
    with pytest.raises(ConfigValidationError, match="must all differ"):
        _good_data(text_column="x", label_column="x", id_column="y")


def test_data_rejects_max_chars_below_min():
    with pytest.raises(ConfigValidationError, match="max_text_chars"):
        _good_data(min_text_chars=100, max_text_chars=10)


def test_data_rejects_invalid_eval_ratio():
    with pytest.raises(ConfigValidationError, match="eval_split_ratio"):
        _good_data(eval_split_ratio=0.5)
    with pytest.raises(ConfigValidationError, match="eval_split_ratio"):
        _good_data(eval_split_ratio=-0.1)


def test_data_rejects_zero_sample_per_class():
    with pytest.raises(ConfigValidationError, match="sample_per_class"):
        _good_data(sample_per_class=0)


def test_data_rejects_duplicate_labels():
    with pytest.raises(ConfigValidationError, match="unique"):
        _good_data(label_order=["A1", "A2", "A1"])


# ---------------------------------------------------------------------------
# ModelConfig / TrainingConfig / RuntimeConfig / ReportingConfig
# ---------------------------------------------------------------------------


def test_model_rejects_unknown_head():
    with pytest.raises(ConfigValidationError, match="head_type"):
        ModelConfig(head_type="mlp")


def test_training_load_best_requires_eval():
    with pytest.raises(ConfigValidationError, match="load_best_model_at_end"):
        TrainingConfig(eval_strategy="no", load_best_model_at_end=True)


def test_training_save_must_match_eval_when_load_best():
    with pytest.raises(ConfigValidationError, match="save_strategy == eval_strategy"):
        TrainingConfig(
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=True,
        )


def test_training_lr_must_be_open_interval():
    with pytest.raises(ConfigValidationError, match="body_learning_rate"):
        TrainingConfig(body_learning_rate=0)
    with pytest.raises(ConfigValidationError, match="head_learning_rate"):
        TrainingConfig(head_learning_rate=1.5)


def test_runtime_device_regex():
    RuntimeConfig(device="cuda:0")
    RuntimeConfig(device="auto")
    with pytest.raises(ConfigValidationError, match="device"):
        RuntimeConfig(device="tpu")


def test_reporting_top_k_non_negative():
    with pytest.raises(ConfigValidationError, match="top_k_examples"):
        ReportingConfig(top_k_examples=-1)


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------


def test_config_cross_section_no_eval_disallows_load_best():
    with pytest.raises(ConfigValidationError, match="load_best_model_at_end"):
        Config(
            data=_good_data(eval_split_ratio=0.0),
            training=TrainingConfig(
                eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True
            ),
        )


def test_config_cross_section_no_eval_forces_eval_strategy_no():
    # When load_best is disabled but eval_strategy is still non-"no",
    # the cross-section check should catch the inconsistency.
    with pytest.raises(ConfigValidationError, match="eval_strategy"):
        Config(
            data=_good_data(eval_split_ratio=0.0),
            training=TrainingConfig(
                eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=False
            ),
        )


def test_config_batch_size_cannot_exceed_sampled_size():
    with pytest.raises(ConfigValidationError, match="exceeds total sampled training size"):
        Config(
            data=_good_data(sample_per_class=2),  # 2 * 6 classes = 12
            training=TrainingConfig(batch_size=32),
        )


def test_config_rejects_unknown_yaml_key(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            data:
              train_files: [fake/a.csv]
              not_a_real_field: true
            """
        ).strip()
    )
    with pytest.raises(ConfigValidationError, match="unknown field"):
        Config.from_yaml(bad)


def test_config_from_sources_layers_overrides(tmp_path: Path):
    y = tmp_path / "cfg.yaml"
    y.write_text(
        textwrap.dedent(
            """
            data:
              train_files: [fake/a.csv]
            training:
              num_epochs: 1
            """
        ).strip()
    )
    cfg = Config.from_sources(
        yaml_path=y,
        flag_overrides={"training.num_epochs": 5},
        dotted_overrides=["data.sample_per_class=16", "model.head_type=setfit_head"],
    )
    assert cfg.training.num_epochs == 5
    assert cfg.data.sample_per_class == 16
    assert cfg.model.head_type == "setfit_head"


def test_override_parses_yaml_value_types(tmp_path: Path):
    y = tmp_path / "cfg.yaml"
    y.write_text("data:\n  train_files: [fake/a.csv]\n")
    cfg = Config.from_sources(
        yaml_path=y,
        dotted_overrides=["training.use_amp=true", "data.label_order=[A,B,C]"],
    )
    assert cfg.training.use_amp is True
    assert cfg.data.label_order == ["A", "B", "C"]


def test_canonical_yaml_hash_is_stable():
    cfg = Config(data=_good_data())
    # same inputs → same hash
    assert model_hash(cfg) == model_hash(cfg)
    # hash is 12 hex chars
    h = model_hash(cfg)
    assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)
    # changing a field changes the hash
    cfg2 = Config(
        data=_good_data(),
        training=TrainingConfig(num_epochs=7),
    )
    assert model_hash(cfg) != model_hash(cfg2)


def test_canonical_yaml_preserves_field_order():
    y = canonical_yaml(Config(data=_good_data()))
    # data must come before model in the canonical form (dataclass order).
    assert y.index("data:") < y.index("model:")


def test_override_rejects_missing_equals():
    with pytest.raises(ConfigValidationError, match="KEY.PATH=VALUE"):
        Config.from_sources(dotted_overrides=["no_equals_sign"])
