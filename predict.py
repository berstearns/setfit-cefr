#!/usr/bin/env python
"""Thin entry point: delegates to setfit_cefr.cli.predict_main.

Usage:
    python predict.py \
        --model models/<hash> \
        --test-files data/norm-EFCAMDAT-test.csv data/norm-KUPA-KEYS.csv data/norm-CELVA-SP.csv

    # With explicit config override:
    python predict.py \
        --model models/<hash> \
        --test-files data/norm-EFCAMDAT-test.csv \
        --config configs/default.yaml \
        --override reporting.top_k_examples=10
"""

from __future__ import annotations

import sys

from setfit_cefr.cli import predict_main


if __name__ == "__main__":
    sys.exit(predict_main())
