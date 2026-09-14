from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from api.schemas.patient import SyntheticPatientProfile


class CaseEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patient: SyntheticPatientProfile = Field(..., description="Multi-modal patient profile to simulate and evaluate")
    clinical_note: Optional[str] = Field(None, description="Optional custom clinical progress note for Stage 04 SLM")


class ModelEvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patient_id: str = Field(..., description="ID of evaluated patient profile")
    stage01_ctcae_toxicity_grade: int = Field(..., description="Stage 01 predicted CTCAE Toxicity Grade (0-4)")
    stage02_recist_status: str = Field(..., description="Stage 02 RECIST Progression status (Progression, Response, Stable)")
    stage04_slm_recommendation: str = Field(..., description="Stage 04 SLM Agent treatment recommendation text")
    safety_hold_triggered: bool = Field(..., description="Whether SLM or safety guard triggered clinical hold")
    safety_violation_flag: bool = Field(..., description="True if an unsafe recommendation was made under high toxicity")
    audit_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed breakdown of rules and flags evaluated")
