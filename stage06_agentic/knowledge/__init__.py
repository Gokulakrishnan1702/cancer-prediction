"""
Knowledge Package for Stage 06 Agentic AI
"""
import os
import json
from typing import Dict, Any
from stage06_agentic.knowledge.oncology_knowledge import OncologyKnowledgeBase

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_nccn_guidelines() -> Dict[str, Any]:
    return OncologyKnowledgeBase.load_nccn()


def load_drug_formulary() -> Dict[str, Any]:
    return OncologyKnowledgeBase.load_formulary()


def load_clinical_trials() -> Dict[str, Any]:
    return OncologyKnowledgeBase.load_trials()


__all__ = [
    "OncologyKnowledgeBase",
    "load_nccn_guidelines",
    "load_drug_formulary",
    "load_clinical_trials"
]
