# setfit-cefr

SetFit-based CEFR classification with a fully YAML-configurable, hyper-reproducible CLI.

- **Training data**: `norm-EFCAMDAT-train.csv` + `norm-EFCAMDAT-remainder.csv`
- **External test sets**: `norm-EFCAMDAT-test.csv`, `norm-KUPA-KEYS.csv`, `norm-CELVA-SP.csv`
- **Model**: [huggingface/setfit](https://github.com/huggingface/setfit) on top of a multilingual Sentence Transformer
- **Reproducibility**: content-addressed output folders (`models/<hash>/`, `predictions/<hash>/`), canonical YAML, file fingerprints, deterministic seeds

---

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) (Python packaging) and [`just`](https://just.systems) (task runner).

```bash
# 0. Install uv & just (one-time, per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh
cargo install just     # or: brew install just

# 1. Create the virtualenv and install the package (editable + dev extras)
just setup

# 2. Link the CSVs into ./data (edit env vars if your paths differ)
just link-data

# 3. Sanity-check the environment
just doctor

# 4. Run the unit tests (no model downloads)
just test

# 5. Smoke run: tiny config, finishes in minutes even on CPU
just train-smoke

# 6. Full train
just train                        # uses configs/default.yaml

# 7. Predict on all three external test sets
just predict models/<model-hash>
```

## Layout

```
setfit-cefr/
├── pyproject.toml           # uv / pip / hatchling project definition
├── Justfile                 # task runner recipes (setup / train / predict / test)
├── .python-version          # 3.11 (pinned for determinism)
├── configs/
│   ├── default.yaml         # full-surface config, production defaults
│   └── smoke.yaml           # tiny config for a <5-min smoke run
├── data/                    # symlinks to the CSVs (created by `just link-data`)
├── train.py                 # thin entry point → setfit_cefr.cli.train_main
├── predict.py               # thin entry point → setfit_cefr.cli.predict_main
├── src/setfit_cefr/
│   ├── config.py            # composable dataclasses with strict post-init validation
│   ├── data.py              # load / clean / sample-per-class / stratified split
│   ├── hashing.py           # model-hash, predict-hash, file fingerprints
│   ├── training.py          # SetFit training pipeline
│   ├── inference.py         # batched predict_proba + metrics per test file
│   ├── reporting.py         # accuracy / macro-F1 / QWK / adjacent-acc + markdown
│   └── cli.py               # shared argparse, YAML + flag + --override merge
└── tests/
    ├── test_config.py       # 25+ validation & override assertions
    ├── test_hashing.py
    └── test_reporting.py
```

## Configuration model

Configs are frozen-at-construction composable dataclasses:

```
Config
├── data:      DataConfig      # files, columns, filters, sampling, eval split
├── model:     ModelConfig     # backbone, head type, multi-target strategy
├── training:  TrainingConfig  # epochs, batch, LRs, seed, eval/save strategies
├── runtime:   RuntimeConfig   # device, output roots, cache dir, workers
├── reporting: ReportingConfig # which metrics + artefacts to emit
├── run_name:       str | None
└── experiment_tag: str
```

Every `__post_init__` performs granular checks:

- `data.train_files` non-empty, unique, and (by default) existing on disk
- `data.{text,label,id}_column` all distinct and non-empty
- `data.label_order` unique non-empty strings
- `data.min_text_chars <= data.max_text_chars`
- `data.eval_split_ratio` in `[0.0, 0.5)`
- `model.head_type` in `{logistic, setfit_head}`
- `training.{body,head}_learning_rate` strictly in `(0, 1)`
- `training.load_best_model_at_end ⇒ eval_strategy == save_strategy ≠ "no"`
- `runtime.device` matches `auto | cpu | cuda | cuda:N`
- Cross-section: `batch_size ≤ sample_per_class × |label_order|`
- Cross-section: `eval_split_ratio == 0 ⇒ load_best_model_at_end == false`

Unknown YAML keys are rejected with a precise dot-pathed error so typos never silently disappear.

## Config overrides

Three layered sources, merged in order (later wins):

1. **YAML**            — `--config configs/default.yaml`
2. **Flag shortcuts**  — e.g. `--epochs 3 --batch-size 32 --seed 7`
3. **Arbitrary path overrides** — `--override training.num_epochs=3 --override data.sample_per_class=32`

Each `--override` value is parsed as YAML, so ints, bools, and lists all work:

```bash
python train.py --config configs/default.yaml \
    --override training.use_amp=true \
    --override data.label_order='[A1,A2,B1,B2,C1,C2]' \
    --override training.num_iterations=40
```

## Output contracts

### `train.py` → `models/<model-hash>/`

```
models/<model-hash>/
├── setfit-model/            # SetFitModel.save_pretrained(...) output (for inference)
├── checkpoints/             # per-epoch checkpoints from the SetFit Trainer
├── config.yaml              # the canonical resolved config (determines the hash)
├── config.hash              # the 12-char model hash
├── labels.json              # ordered CEFR label list
├── train_manifest.json      # rows per split, label distribution, CSV fingerprints
├── training_log.json        # per-epoch/step log from trainer.state.log_history
└── eval_metrics.json        # final held-out metrics
```

The hash is `sha256(canonical_yaml(resolved_config))[:12]`. Identical configs
collapse to the same folder; changing any field (even `experiment_tag`)
produces a new one.

### `predict.py` → `predictions/<predict-hash>/`

```
predictions/<predict-hash>/
├── predictions_probas.json  # per test file: ids, y_true, y_pred, probas
├── report.json              # per test file: accuracy, macro-F1, QWK, adjacent-acc, confusion matrix
├── report.md                # human-readable summary of report.json
├── predict_config.yaml      # resolved config used for prediction
└── predict_manifest.json    # model hash, test-file fingerprints, override blob
```

`predict_hash = sha256(model_hash + sorted(resolved_test_paths) + overrides_yaml)[:12]`,
which means re-running with the same inputs is a cache hit.

## Metrics

- **accuracy**
- **macro-F1**
- **Quadratic Weighted Kappa** — the standard ordinal metric for CEFR
- **adjacent-accuracy** — fraction predicted within ±1 CEFR level
- **confusion matrix** — ordered by `data.label_order`

Metrics are skipped gracefully when a test CSV has no `cefr_level` column
(e.g. unlabelled inference).

## Dry-run / hash preview

```bash
# Print the model hash for a config without training:
python train.py --config configs/default.yaml --dry-run

# Same for prediction (also doesn't load the model):
python predict.py --model models/<hash> --test-files data/norm-EFCAMDAT-test.csv --dry-run
```

## Running on GPU

```bash
just train configs/default.yaml --device cuda --batch-size 32
```

`runtime.device=auto` (the default) picks CUDA if available, else CPU.

## Development

```bash
just test        # unit tests for config validation, hashing, reporting
just lint        # ruff check
just format      # ruff format
just check       # lint + test (CI gate)
```

## License

Apache-2.0. See `pyproject.toml`.
