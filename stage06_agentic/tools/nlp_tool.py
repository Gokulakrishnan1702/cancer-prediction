"""
Stage 06 NLP Tool
Read-only connector to existing Stage 3 NLP clinical triage & named entity extraction.
"""
from typing import Dict, Any
from stage06_agentic.tools.stage_connectors import call_stage3_nlp


class NLPTool:
    """Tool wrapper providing read-only access to Stage 3 NLP model."""

    name: str = "Stage 3 NLP (Urgency & Entity Pipeline)"
    description: str = "Extracts urgency tier, clinical symptoms, and oncologic entities from patient clinical notes."

    @staticmethod
    async def run(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        return await call_stage3_nlp(patient_dict)
