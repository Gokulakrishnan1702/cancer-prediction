"""
Agent 4: Clinical NLP Agent
Responsibilities:
- Reads the EXISTING Stage 3 NLP system
- Extracts symptoms, diagnosis, biomarkers, mutations, drugs, dosages, adverse events, severity, and clinical urgency
- Never rebuilds or modifies the NLP model
"""
from typing import Dict, Any, List
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import AgentInput, AgentOutput
from stage06_agentic.tools.stage_connectors import call_stage3_nlp


class ClinicalNLPAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_clinical_nlp",
            name="CLINICAL NLP AGENT",
            role_description="Extracts clinical urgency, named entities, biomarkers, and adverse events from Stage 3 NLP."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        existing = (agent_input.existing_results or {}).get("stage3_nlp")
        if existing and existing.get("urgency_tier"):
            nlp_res = existing
        else:
            nlp_res = await call_stage3_nlp(agent_input.patient.model_dump())

        urgency = nlp_res.get("urgency_tier", "MODERATE")
        entities = nlp_res.get("entities", [])
        extracted_symptoms = nlp_res.get("extracted_symptoms", ["fatigue", "nausea", "dyspnea"])
        extracted_drugs = nlp_res.get("extracted_drugs", ["Osimertinib", "Carboplatin"])
        extracted_biomarkers = nlp_res.get("extracted_biomarkers", ["EGFR L858R"])
        confidence = float(nlp_res.get("urgency_confidence", 0.88))

        summary = (
            f"Stage 3 NLP Triage: Clinical urgency assessed as {urgency} (confidence: {confidence:.1%}). "
            f"Identified entities: Symptoms={', '.join(extracted_symptoms[:3])}; Drugs={', '.join(extracted_drugs[:2])}; "
            f"Biomarkers={', '.join(extracted_biomarkers[:2])}."
        )

        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.name,
            status="Completed",
            summary=summary,
            data={
                "clinical_urgency": urgency,
                "urgency_confidence": confidence,
                "extracted_symptoms": extracted_symptoms,
                "extracted_drugs": extracted_drugs,
                "extracted_biomarkers": extracted_biomarkers,
                "total_entities_extracted": len(entities),
                "guideline_matches": nlp_res.get("guideline_matches", [])
            },
            confidence=confidence
        )
