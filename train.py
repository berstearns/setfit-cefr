#!/usr/bin/env python
"""Thin entry point: delegates to setfit_cefr.cli.train_main.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --epochs 3 --seed 7
    python train.py --config configs/default.yaml \
        --override training.num_epochs=3 \
        --override data.sample_per_class=32
"""

from __future__ import annotations

import sys

from setfit_cefr.cli import train_main


if __name__ == "__main__":
    sys.exit(train_main())
