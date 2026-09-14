from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class CapstoneWildcardAudit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patient_id: str = Field(..., description="Target wildcard patient identifier (SYNTH_EDGE_020)")
    labs_alt: Optional[float] = Field(None, description="ALT baseline value")
    genomics: list[str] = Field(default_factory=list, description="Genomic alteration profile")
    stage1_toxicity_prediction: int = Field(..., description="Stage 01 predicted toxicity grade")
    slm_decision: str = Field(..., description="Stage 04 SLM treatment decision")
    slm_triggered_safety_hold: bool = Field(..., description="Whether SLM initiated safety hold")
    passed_safety_audit: bool = Field(..., description="Whether safety audit criteria were satisfied")


class AuditReportResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    performance_decay: Dict[str, Any] = Field(..., description="Performance degradation metrics across models")
    cross_modal_safety: Dict[str, Any] = Field(..., description="Safety violation and cross-modal audit statistics")
    capstone_wildcard_audit: CapstoneWildcardAudit = Field(..., description="Detailed audit of capstone edge case SYNTH_EDGE_020")
    wasserstein_kl_divergence: Optional[Dict[str, Any]] = Field(None, description="Wasserstein distance and KL divergence scores")
