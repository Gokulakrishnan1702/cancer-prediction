"""
Stage 06 ML Tool
Read-only connector to existing Stage 1 ML toxicity prediction service.
Strictly reads existing models and outputs; zero retraining or duplication.
"""
from typing import Dict, Any
from stage06_agentic.tools.stage_connectors import call_stage1_ml


class MLTool:
    """Tool wrapper providing read-only access to Stage 1 ML ensemble."""

    name: str = "Stage 1 ML (Calibrated Ensemble)"
    description: str = "Retrieves toxicity risk class, probability, and contributing feature importances from Stage 1."

    @staticmethod
    async def run(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        return await call_stage1_ml(patient_dict)
