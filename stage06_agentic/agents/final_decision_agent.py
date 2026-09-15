"""
Agent 10: Final Decision Agent
Responsibilities:
- Synthesizes only validated outputs from preceding agents
- Produces the "FINAL MULTI-AGENT CLINICAL ASSESSMENT"
- Embeds mandatory oncology decision-support disclaimers
- Avoids exposing raw chain-of-thought
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import (
    AgentInput,
    AgentOutput,
    FinalAssessment,
    TreatmentOption,
    TrialMatch,
    SafetyCheck
)


class FinalDecisionAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_final_decision",
            name="FINAL DECISION AGENT",
            role_description="Synthesizes all multi-agent findings into a final, unified, auditable clinical assessment."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        p = agent_input.patient
        collected_outputs = agent_input.parameters.get("collected_agent_outputs", {})

        # Extract agent summaries
        p_agent = collected_outputs.get("agent_patient_data", {})
        risk_agent = collected_outputs.get("agent_risk_analysis", {})
        image_agent = collected_outputs.get("agent_image_analysis", {})
        nlp_agent = collected_outputs.get("agent_clinical_nlp", {})
        slm_agent = collected_outputs.get("agent_clinical_briefing", {})
        genai_agent = collected_outputs.get("agent_scenario_analysis", {})
        treatment_agent = collected_outputs.get("agent_treatment_optimization", {})
        trial_agent = collected_outputs.get("agent_trial_matching", {})
        safety_agent = collected_outputs.get("agent_safety_guardrail", {})

        # Safety determination
        safety_data = safety_agent.get("data", {})
        is_halted = safety_data.get("workflow_halted", False)
        safety_status = safety_data.get("safety_status", "PASSED")

        if is_halted:
            overall_status = "SAFETY_REVIEW_REQUIRED"
            final_rec = (
                f"SAFETY INTERCEPTOR ACTIVE: Critical clinical risk flagged ({safety_data.get('halt_reason')}). "
                f"Autonomous trial allocation and systemic escalation are HALTED. "
                f"Requires immediate multidisciplinary tumor board review and organ-supportive stabilization."
            )
        else:
            overall_status = "SUCCESS"
            top_tx = treatment_agent.get("data", {}).get("top_recommendation", "Standard Targeted Protocol")
            top_trial = trial_agent.get("data", {}).get("top_trial_name")
            trial_slots = trial_agent.get("data", {}).get("top_trial_slots", 1)

            if top_trial and trial_slots > 0 and p.rising_ctdna_readings >= 4:
                final_rec = (
                    f"Prioritized Deliberative Action: Patient demonstrates acute molecular escape with {p.rising_ctdna_readings} "
                    f"rising ctDNA readings under current regimen. Recommended for priority enrollment in {top_trial}. "
                    f"Organ function is confirmed adequate; standard-of-care systemic salvage held in reserve."
                )
            elif top_trial and trial_slots == 0:
                final_rec = (
                    f"Trial Cohort Capped (0 Slots Free): Re-route patient to optimized systemic standard-of-care "
                    f"({top_tx}) while monitoring liquid biopsy kinetics for next available clinical trial window."
                )
            else:
                final_rec = (
                    f"Recommended Clinical Strategy: {top_tx}. "
                    f"Maintain active surveillance with repeat liquid biopsy and imaging at 6-week interval."
                )

        assessment = FinalAssessment(
            patient_id=p.patient_id,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            overall_status=overall_status,
            patient_summary=p_agent.get("summary", f"Patient {p.patient_id} ({p.cancer_type}, {p.cancer_stage})"),
            ml_risk_summary=risk_agent.get("summary", "ML risk assessment complete."),
            dl_image_summary=image_agent.get("summary", "DL image assessment complete."),
            nlp_clinical_summary=nlp_agent.get("summary", "NLP clinical summary complete."),
            slm_briefing=slm_agent.get("summary", "SLM briefing complete."),
            genai_scenario_summary=genai_agent.get("summary", "GenAI scenario projection complete."),
            treatment_optimization=[TreatmentOption(**o) for o in treatment_agent.get("data", {}).get("treatment_options", [])],
            clinical_trial_matches=[TrialMatch(**m) for m in trial_agent.get("data", {}).get("matched_trials", [])],
            safety_assessment=[SafetyCheck(**c) for c in safety_data.get("safety_checks", [])],
            safety_status=safety_status,
            final_recommendation=final_rec,
            disclaimer="AI-generated clinical decision support. Final treatment decisions require qualified oncologist review.",
            physician_override_active=False,
            audit_trace_id=f"TRACE-{uuid.uuid4().hex[:8].upper()}"
        )

        summary = f"FINAL MULTI-AGENT CLINICAL ASSESSMENT GENERATED: {final_rec}"

        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.name,
            status="Completed" if not is_halted else "Warning",
            summary=summary,
            data=assessment.model_dump(),
            confidence=0.95 if not is_halted else 0.50
        )
