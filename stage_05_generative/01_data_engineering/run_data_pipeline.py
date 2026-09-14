"""
Stage 05 - Data Engineering Pipeline Runner
Executes: Baseline extraction -> Synthetic validation -> SQLite DB ingestion
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if STAGE_ROOT not in sys.path:
    sys.path.insert(0, STAGE_ROOT)

from extract_seeds import extract_baselines
from validate_outputs import validate_synthetic_outputs
from load_to_db import load_to_db


def main():
    print("=" * 60)
    print(">>> STAGE 05: DATA ENGINEERING PIPELINE EXECUTION <<<")
    print("=" * 60)

    # 1. Extract baseline distributions
    print("\n--- Step 1: Extract Baseline Distributions ---")
    extract_baselines()

    # 2. Validate Synthetic Outputs
    print("\n--- Step 2: Validate Schema & Medical Bounds ---")
    validate_synthetic_outputs()

    # 3. Ingest into SQLite Database
    print("\n--- Step 3: Ingest into SQLite Database ---")
    load_to_db()

    print("\n[DATA ENGINEERING COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
