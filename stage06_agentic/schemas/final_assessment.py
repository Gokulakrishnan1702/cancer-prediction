"""
Final Assessment & Decision Trace Schemas
Structured models representing the end-to-end multi-agent clinical consensus.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from stage06_agentic.schemas.treatment import TreatmentOption
from stage06_agentic.schemas.trial import TrialMatch
from stage06_agentic.schemas.safety import SafetyCheck


class DecisionTrace(BaseModel):
    step_number: int
    agent_name: str
    tool_or_api: str
    status: str
    timestamp: str
    reasoning_summary: str
    key_findings: List[str] = Field(default_factory=list)


class FinalAssessment(BaseModel):
    patient_id: str
    timestamp: str
    overall_status: str  # SUCCESS, SAFETY_REVIEW_REQUIRED, OVERRIDDEN, ABORTED
    patient_summary: str
    ml_risk_summary: str
    dl_image_summary: str
    nlp_clinical_summary: str
    slm_briefing: str
    genai_scenario_summary: str
    treatment_optimization: List[TreatmentOption] = Field(default_factory=list)
    clinical_trial_matches: List[TrialMatch] = Field(default_factory=list)
    safety_assessment: List[SafetyCheck] = Field(default_factory=list)
    safety_status: str = "PASSED"  # PASSED, REQUIRES_REVIEW, CRITICAL_HALT, SAFETY REVIEW REQUIRED
    final_recommendation: str
    disclaimer: str = (
        "AI-generated clinical decision support. Final treatment decisions require qualified oncologist review."
    )
    uncertainty_limitations: str = (
        "Recommendations derived from multi-stage AI models and verified guidelines. "
        "Clinical trials subject to real-time site verification and IRB confirmation."
    )
    physician_override_active: bool = False
    physician_override_notes: Optional[str] = None
    audit_trace_id: str
