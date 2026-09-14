"""
Stage 05 - EDA & Audit Engineering Package
Handles latent profiling, Wasserstein/KL distribution audits, text token analytics, and unified EDA reporting.
"""

from .eda_latent_profiling import run_latent_profiling
from .eda_distribution_audit import run_distribution_audit
from .eda_text_analytics import run_text_analytics
from .generate_eda_report import generate_master_report

__all__ = [
    "run_latent_profiling",
    "run_distribution_audit",
    "run_text_analytics",
    "generate_master_report",
]
