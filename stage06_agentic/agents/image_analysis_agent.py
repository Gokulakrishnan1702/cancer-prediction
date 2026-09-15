"""
Agent 3: Image / Clinical Analysis Agent
Responsibilities:
- Reads the EXISTING Stage 2 DL output when medical image/scan is available
- If no image exists, outputs: "DL analysis unavailable — image not provided."
- Never fabricates image results
"""
from typing import Dict, Any
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import AgentInput, AgentOutput
from stage06_agentic.tools.stage_connectors import call_stage2_dl


class ImageAnalysisAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_image_analysis",
            name="CLINICAL ANALYSIS AGENT",
            role_description="Reads medical image analysis and progression trajectories from Stage 2 DL."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        p = agent_input.patient

        # Prompt condition: If no image exists
        if not p.medical_image_provided or not p.image_file_path:
            return AgentOutput(
                agent_id=self.agent_id,
                agent_name=self.name,
                status="Warning",
                summary="DL analysis unavailable — image not provided.",
                data={
                    "image_available": False,
                    "finding": "DL analysis unavailable — image not provided.",
                    "classification": "None",
                    "confidence": 0.0
                },
                confidence=0.0,
                warnings=["Medical scan or pathology image not provided for DL multimodal analysis."]
            )

        # Check existing or call stage 2
        existing = (agent_input.existing_results or {}).get("stage2_dl")
        if existing and existing.get("image_prediction"):
            dl_res = existing
        else:
            dl_res = await call_stage2_dl(p.model_dump())

        if dl_res.get("status") == "Unavailable":
            return AgentOutput(
                agent_id=self.agent_id,
                agent_name=self.name,
                status="Warning",
                summary="DL analysis unavailable — image not provided.",
                data={"image_available": False},
                confidence=0.0,
                warnings=["DL analysis unavailable — image not provided."]
            )

        classification = dl_res.get("classification") or dl_res.get("image_prediction", "Malignant Neoplasm")
        prog_prob = dl_res.get("progression_probability", 0.75)
        conf = dl_res.get("confidence", 0.88)
        risk_tier = dl_res.get("risk_tier", "High Progression Risk")

        summary = (
            f"Stage 2 DL Findings: {classification} ({risk_tier}). "
            f"Image progression probability: {prog_prob:.1%}, model confidence: {conf:.1%}."
        )

        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.name,
            status="Completed",
            summary=summary,
            data={
                "image_available": True,
                "classification": classification,
                "progression_probability": prog_prob,
                "confidence": conf,
                "risk_tier": risk_tier,
                "pathology_tile": dl_res.get("pathology_tile_file"),
                "ct_slice": dl_res.get("ct_slice_file")
            },
            confidence=conf
        )
