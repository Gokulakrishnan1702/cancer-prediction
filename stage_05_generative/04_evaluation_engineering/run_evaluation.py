"""
Stage 05 - Evaluation Engineering Pipeline Runner
Executes: Multi-model inference -> Decay metrics -> Safety auditing -> Report export
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if STAGE_ROOT not in sys.path:
    sys.path.insert(0, STAGE_ROOT)

from evaluate_models import run_stress_test


def main():
    print("=" * 60)
    print(">>> STAGE 05: MODEL STRESS-TEST & EVALUATION PIPELINE <<<")
    print("=" * 60)

    # Run stress test
    run_stress_test()

    print("\n[EVALUATION & SAFETY ENGINEERING COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
