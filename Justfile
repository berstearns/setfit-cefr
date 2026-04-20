# setfit-cefr task runner. Requires `just` and `uv` installed.
# Usage: `just` to list recipes, `just <recipe> [args]` to run.

set shell := ["bash", "-euo", "pipefail", "-c"]
set dotenv-load := true

default:
    @just --list

# --- Environment setup --------------------------------------------------------

# Create a pinned virtualenv with uv and install the project (editable + dev).
setup:
    uv sync --extra dev
    @echo "Done. Activate with: source .venv/bin/activate"

# Verify the environment can import the package and setfit.
doctor:
    uv run python -c "import setfit, setfit_cefr; print('setfit', setfit.__version__)"

# --- Data -------------------------------------------------------------------

# Symlink the 5 CEFR CSVs from a single user-supplied directory into ./data/.
# Point DATA_SRC at a folder that contains the files named exactly as below.
# The individual *_PATH env vars let you override per-file if your layout
# splits train/test across directories.
#
#   export DATA_SRC=/path/to/your/csvs
#   just link-data
link-data:
    @if [ -z "${DATA_SRC:-}" ] && \
        [ -z "${EFCAMDAT_TRAIN_PATH:-}" ] && \
        [ -z "${EFCAMDAT_REMAINDER_PATH:-}" ] && \
        [ -z "${TEST_DIR:-}" ]; then \
        echo "Error: set DATA_SRC to the directory holding the 5 CSVs, e.g."; \
        echo "    export DATA_SRC=/path/to/csvs && just link-data"; \
        echo "Or set each of EFCAMDAT_TRAIN_PATH, EFCAMDAT_REMAINDER_PATH, TEST_DIR."; \
        exit 1; \
    fi
    mkdir -p data
    @EFCAMDAT_TRAIN_PATH="${EFCAMDAT_TRAIN_PATH:-${DATA_SRC}/norm-EFCAMDAT-train.csv}"; \
     EFCAMDAT_REMAINDER_PATH="${EFCAMDAT_REMAINDER_PATH:-${DATA_SRC}/norm-EFCAMDAT-remainder.csv}"; \
     TEST_DIR="${TEST_DIR:-${DATA_SRC}}"; \
     ln -sf "$EFCAMDAT_TRAIN_PATH"                data/norm-EFCAMDAT-train.csv; \
     ln -sf "$EFCAMDAT_REMAINDER_PATH"            data/norm-EFCAMDAT-remainder.csv; \
     ln -sf "$TEST_DIR/norm-EFCAMDAT-test.csv"    data/norm-EFCAMDAT-test.csv; \
     ln -sf "$TEST_DIR/norm-KUPA-KEYS.csv"        data/norm-KUPA-KEYS.csv; \
     ln -sf "$TEST_DIR/norm-CELVA-SP.csv"         data/norm-CELVA-SP.csv
    @ls -l data/

# --- Train / predict --------------------------------------------------------

# Train with a config. Extra args are forwarded, e.g.
#   just train configs/default.yaml --override training.num_epochs=2
train config="configs/default.yaml" *args="":
    uv run python train.py --config {{config}} {{args}}

# Quick smoke run: small sample-per-class + 1 epoch on CPU.
train-smoke:
    uv run python train.py --config configs/smoke.yaml

# Predict on the three external test sets using a trained model folder.
# Usage: just predict models/<hash>
predict model *args="":
    uv run python predict.py \
        --model {{model}} \
        --test-files data/norm-EFCAMDAT-test.csv data/norm-KUPA-KEYS.csv data/norm-CELVA-SP.csv \
        {{args}}

# --- Quality gates ----------------------------------------------------------

test:
    uv run pytest tests/ -v

lint:
    uv run ruff check src tests train.py predict.py

format:
    uv run ruff format src tests train.py predict.py

check: lint test

# --- Housekeeping -----------------------------------------------------------

# Remove transient artifacts (keeps data/ and configs/).
clean:
    rm -rf .pytest_cache .ruff_cache __pycache__ src/setfit_cefr/__pycache__ tests/__pycache__

# DANGER: also deletes trained models and predictions.
clean-all: clean
    rm -rf models predictions logs
