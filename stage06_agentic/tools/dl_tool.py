"""
Stage 06 DL Tool
Read-only connector to existing Stage 2 DL imaging prediction service.
Strictly handles missing images safely without hallucinating fake scan data.
"""
from typing import Dict, Any
from stage06_agentic.tools.stage_connectors import call_stage2_dl


class DLTool:
    """Tool wrapper providing read-only access to Stage 2 DL imaging."""

    name: str = "Stage 2 DL (MultimodalLSTM)"
    description: str = "Retrieves radiological lesion classification and progression tier from Stage 2 DL."

    @staticmethod
    async def run(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        return await call_stage2_dl(patient_dict)
