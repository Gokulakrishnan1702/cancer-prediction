"""
Stage 05 - Evaluation & Safety Engineering Package
Evaluates downstream model degradation, flags cross-modal safety violations, and audits capstone edge cases.
"""

from .harness.inference_harness import CDSSEvaluator
from .harness.metrics import compute_decay_metrics, compute_safety_metrics
from .evaluate_models import run_stress_test

__all__ = [
    "CDSSEvaluator",
    "compute_decay_metrics",
    "compute_safety_metrics",
    "run_stress_test",
]
