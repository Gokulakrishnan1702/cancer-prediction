"""
Stage 05 - GenAI Engineering Pipeline Runner
Executes: Tabular VAE training -> Multi-modal scenario generation -> 20 OOD edge cases
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if STAGE_ROOT not in sys.path:
    sys.path.insert(0, STAGE_ROOT)

from generate_edge_cases import generate_20_edge_cases


def main():
    print("=" * 60)
    print(">>> STAGE 05: GEN AI MODELING & SYNTHESIS PIPELINE <<<")
    print("=" * 60)

    # Generate 20 multi-modal out-of-distribution scenarios
    print("\n--- Step 1: Synthesizing Multi-Modal OOD Patient Profiles ---")
    generate_20_edge_cases(num_cases=20, seed=42)

    print("\n[GEN AI ENGINEERING COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
