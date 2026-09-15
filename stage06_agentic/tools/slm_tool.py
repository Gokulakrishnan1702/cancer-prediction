"""
Stage 06 SLM Tool
Read-only connector to existing Stage 4 SLM briefing service.
Provides concise physician-oriented reasoning without exposing raw chain-of-thought.
"""
from typing import Dict, Any, Optional
from stage06_agentic.tools.stage_connectors import call_stage4_slm


class SLMTool:
    """Tool wrapper providing read-only access to Stage 4 SLM reasoning."""

    name: str = "Stage 4 SLM (Concise Reasoning Core)"
    description: str = "Synthesizes concise multi-stage clinical briefing via Stage 4 SLM."

    @staticmethod
    async def run(
        patient_dict: Dict[str, Any],
        stage1_res: Optional[Dict[str, Any]] = None,
        stage2_res: Optional[Dict[str, Any]] = None,
        stage3_res: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return await call_stage4_slm(patient_dict, stage1_res, stage2_res, stage3_res)
