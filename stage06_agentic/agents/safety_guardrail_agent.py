"""
Agent 9: Safety / Guardrail Agent
Responsibilities:
- Intercepts contraindications, organ toxicity, contradictory findings, and dangerous drug interactions
- If a severe/critical clinical risk is detected:
  STOPS WORKFLOW and triggers "SAFETY REVIEW REQUIRED"
"""
from typing import Dict, Any, List
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import AgentInput, AgentOutput, SafetyCheck
from stage06_agentic.tools.formulary_lookup_tool import evaluate_clinical_safety


class SafetyGuardrailAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_safety_guardrail",
            name="SAFETY AGENT",
            role_description="Performs multi-dimensional clinical safety verification across organ toxicity, drug contraindications, and cross-stage contradictions."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        p = agent_input.patient
        safety_checks: List[SafetyCheck] = evaluate_clinical_safety(
            cancer_type=p.cancer_type,
            treatment_name=p.treatment_name,
            creatinine=p.creatinine,
            bilirubin=p.bilirubin,
            alt=p.ALT,
            ast=p.AST,
            platelets=p.platelet_count,
            spo2=p.spo2
        )

        critical_failures = [c for c in safety_checks if not c.passed and c.severity == "Critical"]
        warnings = [c for c in safety_checks if not c.passed and c.severity == "Warning"]

        # Prompt condition: If serious risk is detected: STOP WORKFLOW -> "SAFETY REVIEW REQUIRED"
        if critical_failures:
            status = "Failed"  # Halts workflow
            action_status = "SAFETY REVIEW REQUIRED"
            summary = (
                f"SAFETY INTERCEPTOR HALTED: Critical toxicity violation detected ({critical_failures[0].check_name}). "
                f"{critical_failures[0].clinical_finding} WORKFLOW STOPPED. SAFETY REVIEW REQUIRED."
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
                "safety_checks": [c.model_dump() for c in safety_checks],
                "safety_status": action_status,
                "critical_violations_count": len(critical_failures),
                "warnings_count": len(warnings),
                "workflow_halted": len(critical_failures) > 0,
                "halt_reason": critical_failures[0].clinical_finding if critical_failures else None
            },
            confidence=1.0 if not critical_failures else 0.4,
            warnings=[c.clinical_finding for c in warnings + critical_failures]
        )
