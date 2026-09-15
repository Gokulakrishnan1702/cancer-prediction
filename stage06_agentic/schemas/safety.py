"""
Safety and Guardrail Schemas
Models for organ toxicity safety checks, contraindications, and interlocks.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class SafetyCheck(BaseModel):
    check_name: str
    category: str  # Hepatic, Renal, Hematologic, Pulmonary, Biomarker-Mismatch, Drug-Interaction, Data-Consistency
    passed: bool
    severity: str  # Critical, Warning, Normal
    clinical_finding: str
    action_required: str


class SafetyAssessment(BaseModel):
    safety_status: str  # PASSED, CAUTION_REQUIRED, SAFETY REVIEW REQUIRED
    critical_violations_count: int = 0
    warnings_count: int = 0
    workflow_halted: bool = False
    halt_reason: Optional[str] = None
    safety_checks: List[SafetyCheck] = Field(default_factory=list)
