"""
Stage 05 - EDA Engineering Pipeline Runner
Executes: Latent profiling -> Distribution audit -> Text analytics -> Master report compilation
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if STAGE_ROOT not in sys.path:
    sys.path.insert(0, STAGE_ROOT)

from eda_latent_profiling import run_latent_profiling
from eda_distribution_audit import run_distribution_audit
from eda_text_analytics import run_text_analytics
from generate_eda_report import generate_master_report


def main():
    print("=" * 60)
    print(">>> STAGE 05: EDA & AUDIT ENGINEERING PIPELINE EXECUTION <<<")
    print("=" * 60)

    # 1. Latent Profiling
    print("\n--- Step 1: Latent Space Profiling ---")
    run_latent_profiling()

    # 2. Distribution Audit
    print("\n--- Step 2: Distribution Divergence Audit (Wasserstein & KL) ---")
    run_distribution_audit()

    # 3. Text & Token Analytics
    print("\n--- Step 3: EHR Clinical Notes Token Analytics ---")
    run_text_analytics()

    # 4. Master Report Compilation
    print("\n--- Step 4: Compile Master EDA Report ---")
    generate_master_report()

    print("\n[EDA & AUDIT ENGINEERING COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
