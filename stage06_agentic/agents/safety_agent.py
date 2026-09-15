"""
Agent 6 / Role 3: Safety / Guardrail Agent
Responsibilities:
- Intercepts contraindications, organ toxicities, contradictory findings, and dangerous drug interactions
- If a severe/critical clinical risk is detected:
  STOPS WORKFLOW and triggers "SAFETY REVIEW REQUIRED"
"""
from typing import Dict, Any, List
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import AgentInput, AgentOutput, SafetyCheck, SafetyAssessment
from stage06_agentic.safety.safety_guardrails import evaluate_clinical_safety_guardrails


class SafetyAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_safety_guardrail",
            name="SAFETY AGENT",
            role_description="Performs multi-dimensional clinical safety verification across organ toxicity, drug contraindications, and cross-stage contradictions."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        p = agent_input.patient
        treatment_options = agent_input.parameters.get("treatment_options", [])
        matched_trials = agent_input.parameters.get("matched_trials", [])

        assessment: SafetyAssessment = evaluate_clinical_safety_guardrails(
            patient=p,
            treatment_options=treatment_options,
            matched_trials=matched_trials,
            existing_results=agent_input.existing_results
        )

        critical_failures = [c for c in assessment.safety_checks if not c.passed and c.severity == "Critical"]
        warnings = [c for c in assessment.safety_checks if not c.passed and c.severity == "Warning"]

        # Prompt condition: If serious risk is detected: STOP WORKFLOW -> "SAFETY REVIEW REQUIRED"
        if assessment.workflow_halted or critical_failures:
            status = "Failed"  # Halts workflow
            action_status = "SAFETY REVIEW REQUIRED"
            summary = (
                f"SAFETY INTERCEPTOR HALTED: Critical clinical violation detected ({critical_failures[0].check_name if critical_failures else 'Organ Safety Breach'}). "
                f"{assessment.halt_reason} WORKFLOW STOPPED. SAFETY REVIEW REQUIRED."
            )
        elif warnings:
            status = "Warning"
            action_status = "CAUTION_REQUIRED"
            summary = (
                f"Safety Audit Passed with Caution: {len(warnings)} warning(s) flagged ({warnings[0].check_name}). "
                f"Requires clinical dosage modification and physician monitoring."
            )
        else:
            status = "Completed"
            action_status = "PASSED"
            summary = "Comprehensive Clinical Safety Review Passed: All organ function, hematologic, and formulary safety floors verified."

        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.name,
            status=status,
            summary=summary,
            data={
                "safety_checks": [c.model_dump() for c in assessment.safety_checks],
                "safety_status": action_status,
                "critical_violations_count": len(critical_failures),
                "warnings_count": len(warnings),
                "workflow_halted": assessment.workflow_halted or len(critical_failures) > 0,
                "halt_reason": assessment.halt_reason
            },
            confidence=1.0 if not critical_failures else 0.4,
            warnings=[c.clinical_finding for c in warnings + critical_failures]
        )


SafetyGuardrailAgent = SafetyAgent
