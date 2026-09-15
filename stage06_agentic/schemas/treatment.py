"""
Treatment Schemas
Models for oncology candidate regimens, expected benefit, potential toxicities,
and patient-specific considerations.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class TreatmentOption(BaseModel):
    regimen_name: str
    category: str  # Targeted Therapy, Precision Clinical Trial, Standard-of-Care Chemotherapy, Immunotherapy
    expected_benefit: str
    potential_toxicity: str
    renal_hepatic_suitability: str
    priority_rank: int
    reasons_for_consideration: List[str]
    safety_warnings: List[str]
    evidence_source: str = "NCCN Guidelines Version 2.2024 / FDA Oncology Package Insert"


class TreatmentStrategy(BaseModel):
    patient_id: str
    cancer_type: str
    cancer_stage: str
    candidate_options: List[TreatmentOption] = Field(default_factory=list)
    top_recommendation: Optional[str] = None
    disclaimer: str = "AI-generated clinical decision support — requires qualified oncologist review."
