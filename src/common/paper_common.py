"""Common constants used by the paper result builders.

This module centralizes the mainline result paths, fixed paper asset set,
selected control variables, and the four trade-flow variables used by the
current thesis tables and figures.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import BEST8_CONTROL_COLUMNS, MODELS_DIR, OUTPUTS_DIR, PAPER_ASSETS  # noqa: E402
from src.common.paths import ensure_dir  # noqa: E402


BEST8_ASSETS = PAPER_ASSETS.copy()
BEST8_CONTROLS = BEST8_CONTROL_COLUMNS.copy()
TRADEFLOW4_COLUMNS = [
    "IND_SECTOR_TV_ene_norm",
    "INS_SECTOR_TV_ene_norm",
    "ITVvar",
    "ITVvar_x_dolsha",
]

MAINLINE_ROOT = OUTPUTS_DIR
VISUALIZATION_DIR = MAINLINE_ROOT / "visualizations"

MAINLINE_CLASSIFICATION_OUTPUT_DIR = MAINLINE_ROOT / "classification"
MAINLINE_SUMMARY_DIR = MAINLINE_ROOT / "summary"
MAINLINE_CLASSIFICATION_MODEL_DIR = MODELS_DIR / "classification"


def ensure_best8_dirs() -> None:
    """Ensure all mainline result directories used by step 11 exist."""

    for path in [
        MAINLINE_ROOT,
        VISUALIZATION_DIR,
        MAINLINE_CLASSIFICATION_OUTPUT_DIR,
        MAINLINE_SUMMARY_DIR,
        MAINLINE_CLASSIFICATION_MODEL_DIR,
    ]:
        ensure_dir(path)
