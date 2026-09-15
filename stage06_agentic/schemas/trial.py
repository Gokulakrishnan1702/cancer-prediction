"""
Clinical Trial Schemas
Models for oncology clinical trial matching, eligibility evaluation, and slot allocations.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class TrialMatch(BaseModel):
    trial_id: str
    trial_name: str
    phase: str
    target_mutation: str
    matching_score: float
    eligibility_status: str  # Fully Eligible — Slot Available, Provisionally Eligible, Ineligible, Eligible — Cohort Capped
    open_slots: int
    matched_criteria: List[str]
    missing_criteria: List[str]
    data_source: str = "VERIFIED ONCOLOGY TRIAL REGISTRY"  # or DEMO / MOCK TRIAL DATA
    verification_status: str = "VERIFIED"


class TrialSearchFilter(BaseModel):
    cancer_type: str
    cancer_stage: str
    biomarker: str
    age: float
    creatinine: float
    alt: float
    ast: float
    platelets: float
    rising_ctdna_readings: int
    ctdna_level: float
