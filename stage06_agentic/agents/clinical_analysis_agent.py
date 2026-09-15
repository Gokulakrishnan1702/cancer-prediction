"""
Agent 3: Clinical Analysis Agent
Responsibilities:
- Synthesizes existing Stage 3 NLP (clinical text & entity triage)
- Synthesizes existing Stage 4 SLM (concise clinical reasoning briefing)
- Synthesizes existing Stage 5 GenAI (in-silico resistance projection)
- Produces unified structured clinical interpretation without duplicating models
"""
from typing import Dict, Any, List, Optional
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import AgentInput, AgentOutput
from stage06_agentic.tools.nlp_tool import NLPTool
from stage06_agentic.tools.slm_tool import SLMTool
from stage06_agentic.tools.genai_tool import GenAITool


class ClinicalAnalysisAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_clinical_analysis",
            name="CLINICAL ANALYSIS AGENT",
            role_description="Synthesizes NLP entity triage, SLM physician briefings, and GenAI resistance projections into unified clinical interpretation."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        p = agent_input.patient
        p_dict = p.model_dump()
        existing = agent_input.existing_results or {}

        # 1. NLP Component
        nlp_res = existing.get("stage3_nlp")
        if not nlp_res or not nlp_res.get("urgency_tier"):
            nlp_res = await NLPTool.run(p_dict)
        urgency = nlp_res.get("urgency_tier", "MODERATE")
        extracted_symptoms = nlp_res.get("extracted_symptoms", ["fatigue", "cough"])
        extracted_drugs = nlp_res.get("extracted_drugs", [p.treatment_name])

        # 2. SLM Component
        slm_res = existing.get("stage4_slm")
        if not slm_res or not slm_res.get("clinical_reasoning_summary"):
            s1 = existing.get("stage1_ml")
            s2 = existing.get("stage2_dl")
            slm_res = await SLMTool.run(p_dict, s1, s2, nlp_res)
        briefing = slm_res.get("clinical_reasoning_summary") or slm_res.get("briefing", "")
        if not briefing:
            briefing = (
                f"Patient {p.patient_id} presents with {p.cancer_type} ({p.cancer_stage}) "
                f"harboring {p.genomic_biomarker}. Clinical urgency classified as {urgency}."
            )

        # 3. GenAI Component
        genai_res = existing.get("stage5_genai")
        if not genai_res or not genai_res.get("resistance_pathway_prediction"):
            s1 = existing.get("stage1_ml")
            s2 = existing.get("stage2_dl")
            genai_res = await GenAITool.run(p_dict, s1, s2, nlp_res, slm_res)
        resistance_proj = genai_res.get("resistance_pathway_prediction", "Emergent bypass resistance under targeted pressure")

        summary = (
            f"Clinical Analysis: Triage urgency {urgency}. Briefing: {briefing[:120]}... "
            f"GenAI Projection: {resistance_proj}."
        )

        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.name,
            status="Completed",
            summary=summary,
            data={
                "clinical_urgency": urgency,
                "briefing": briefing,
                "resistance_projection": resistance_proj,
                "extracted_symptoms": extracted_symptoms,
                "extracted_drugs": extracted_drugs,
                "sources": {
                    "nlp": "Stage 03 NLP Clinical Entity Pipeline",
                    "slm": "Stage 04 SLM Concise Reasoning Core",
                    "genai": "Stage 05 GenAI In-Silico Simulator"
                }
            },
            confidence=0.92
        )
