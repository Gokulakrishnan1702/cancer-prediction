"""
STAGE 06: AGENTIC AI — MODULE RUNNER
Wrapper around root run_stage_06.py.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_stage_06 import main

if __name__ == "__main__":
    main()
