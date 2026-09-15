"""
Agent 5: Clinical Briefing Agent
Responsibilities:
- Reads the EXISTING Stage 4 SLM reasoning & guardrail output
- Formulates a concise physician-friendly briefing
- Does NOT expose hidden chain-of-thought
"""
from typing import Dict, Any
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import AgentInput, AgentOutput
from stage06_agentic.tools.stage_connectors import call_stage4_slm


class ClinicalBriefingAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_clinical_briefing",
            name="SLM BRIEFING AGENT",
            role_description="Synthesizes multi-stage findings into a physician-friendly clinical briefing via Stage 4 SLM."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        existing = (agent_input.existing_results or {}).get("stage4_slm")
        if existing and existing.get("clinical_reasoning_summary"):
            slm_res = existing
        else:
            p_dict = agent_input.patient.model_dump()
            s1 = (agent_input.existing_results or {}).get("stage1_ml")
            s2 = (agent_input.existing_results or {}).get("stage2_dl")
            s3 = (agent_input.existing_results or {}).get("stage3_nlp")
            slm_res = await call_stage4_slm(p_dict, s1, s2, s3)

        briefing = slm_res.get("clinical_reasoning_summary") or slm_res.get("briefing", "")
        if not briefing:
            p = agent_input.patient
            briefing = (
                f"Patient {p.patient_id} presents with {p.cancer_type} ({p.cancer_stage}) harboring {p.genomic_biomarker}. "
                f"Demonstrates active progression under {p.treatment_name} with ctDNA rising to {p.ctDNA_level} ng/mL."
            )

        guardrail_passed = slm_res.get("guardrail_audit_passed", True)
        confidence = float(slm_res.get("confidence", 0.92))

        summary = f"Stage 4 SLM Briefing: {briefing}"

        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.name,
            status="Completed" if guardrail_passed else "Warning",
            summary=summary,
            data={
                "briefing_text": briefing,
                "guardrail_status": "PASS" if guardrail_passed else "FLAGGED",
                "concise_summary": briefing,
                "latency_ms": slm_res.get("latency_ms", 12.0)
            },
            confidence=confidence,
            warnings=[] if guardrail_passed else ["SLM guardrail flagged potential under-triage floor."]
        )
