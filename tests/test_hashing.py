from __future__ import annotations

from pathlib import Path

import pytest

from setfit_cefr.config import (
    Config,
    DataConfig,
    set_file_existence_checks,
)
from setfit_cefr.hashing import file_fingerprint, model_hash, predict_hash


@pytest.fixture(autouse=True)
def _no_fs_checks():
    set_file_existence_checks(False)
    yield
    set_file_existence_checks(True)


def _cfg() -> Config:
    return Config(data=DataConfig(train_files=["fake/a.csv"]))


def test_predict_hash_stable_across_reorder():
    m = model_hash(_cfg())
    a = predict_hash(m, ["a.csv", "b.csv"], "")
    b = predict_hash(m, ["b.csv", "a.csv"], "")
    assert a == b


def test_predict_hash_changes_with_files():
    m = model_hash(_cfg())
    a = predict_hash(m, ["a.csv"], "")
    b = predict_hash(m, ["a.csv", "b.csv"], "")
    assert a != b


def test_predict_hash_changes_with_overrides():
    m = model_hash(_cfg())
    a = predict_hash(m, ["a.csv"], "")
    b = predict_hash(m, ["a.csv"], "training.seed=1")
    assert a != b


def test_file_fingerprint(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello")
    fp = file_fingerprint(p)
    assert fp["size_bytes"] == 5
    assert fp["sha256"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
