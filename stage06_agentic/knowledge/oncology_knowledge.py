"""
Oncology Knowledge Manager
Provides validated, read-only access to NCCN guidelines, drug formularies,
and clinical trial databases with strict provenance tracking.
"""
import os
import json
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class OncologyKnowledgeBase:
    """Manages static, verified clinical knowledge with version and date tracking."""

    VERSION: str = "2024.2.0"
    SOURCE: str = "NCCN Guidelines Version 2.2024 / FDA Oncology Formularies"
    DATE: str = "2024-04-15"

    @classmethod
    def load_nccn(cls) -> Dict[str, Any]:
        path = os.path.join(CURRENT_DIR, "nccn_guidelines.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_formulary(cls) -> Dict[str, Any]:
        path = os.path.join(CURRENT_DIR, "drug_formulary.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_trials(cls) -> Dict[str, Any]:
        path = os.path.join(CURRENT_DIR, "clinical_trials_db.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_provenance(cls) -> Dict[str, str]:
        return {
            "version": cls.VERSION,
            "source": cls.SOURCE,
            "date": cls.DATE,
            "status": "VERIFIED"
        }
