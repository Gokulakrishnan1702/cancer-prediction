from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator


class OrganImpairment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    alt_u_l: Optional[float] = Field(None, description="Alanine Aminotransferase level (U/L)")
    ast_u_l: Optional[float] = Field(None, description="Aspartate Aminotransferase level (U/L)")
    total_bilirubin_mg_dl: Optional[float] = Field(None, description="Total Bilirubin (mg/dL)")
    creatinine_mg_dl: Optional[float] = Field(None, description="Serum Creatinine (mg/dL)")

    @field_validator("alt_u_l", "ast_u_l", "total_bilirubin_mg_dl", "creatinine_mg_dl")
    @classmethod
    def validate_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Biomarker values must be non-negative.")
        return v


class TrajectoryPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timepoint: Union[str, int] = Field(..., description="Sequence point identifier (e.g. T0, T1 or 0, 1)")
    ctdna_vaf: float = Field(..., description="Circulating tumor DNA Variant Allele Frequency [0.0, 1.0]")
    tumor_vol_cm3: float = Field(..., description="Tumor volume in cm3 (strictly > 0.0)")

    @field_validator("ctdna_vaf")
    @classmethod
    def validate_vaf_bounds(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"ctdna_vaf must be strictly within [0.0, 1.0], got {v}")
        return v

    @field_validator("tumor_vol_cm3")
    @classmethod
    def validate_tumor_vol(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"tumor_vol_cm3 must be strictly > 0.0, got {v}")
        return v


class SyntheticPatientProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    synthetic_patient_id: str = Field(..., description="Unique synthetic patient identifier (e.g. SYNTH_EDGE_001)")
    is_ood: bool = Field(default=False, description="Flag indicating if the profile represents an Out-of-Distribution scenario")
    organ_impairment_baseline: OrganImpairment = Field(..., description="Baseline organ impairment and liver panel values")
    baseline_genomics: List[str] = Field(default_factory=list, description="List of baseline genomic mutations and variants")
    trajectory_t0_to_t4: List[TrajectoryPoint] = Field(default_factory=list, description="Array of longitudinal tumor dynamics from T0 to T4")
    toxicity_grade: Optional[int] = Field(None, description="CTCAE toxicity grade ground-truth if present")
    is_severe_dili: Optional[bool] = Field(None, description="Severe Drug-Induced Liver Injury flag")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata and clinical scenario context")

    @field_validator("synthetic_patient_id")
    @classmethod
    def validate_patient_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("synthetic_patient_id cannot be empty")
        return v.strip()


class PatientPaginationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total: int = Field(..., description="Total synthetic records matching filter")
    limit: int = Field(..., description="Number of records returned")
    offset: int = Field(..., description="Pagination offset index")
    patients: List[SyntheticPatientProfile] = Field(..., description="List of patient profiles")


class MultiModalPatientResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patient: SyntheticPatientProfile = Field(..., description="Structured synthetic patient profile")
    clinical_note: Optional[str] = Field(None, description="Unstructured EHR clinical progress note from ChromaDB")
    chroma_document_id: Optional[str] = Field(None, description="Vector store document ID for note")
