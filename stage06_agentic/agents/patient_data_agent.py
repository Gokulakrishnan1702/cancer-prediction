"""
Agent 1: Patient Data Agent
Responsibilities:
- Validate patient context
- Check missing fields
- Normalize information
- Prepare structured patient context
"""
from typing import Dict, Any, List
from stage06_agentic.agents.base_agent import BaseClinicalAgent
from stage06_agentic.schemas.agent_schemas import AgentInput, AgentOutput, PatientContext


class PatientDataAgent(BaseClinicalAgent):
    def __init__(self):
        super().__init__(
            agent_id="agent_patient_data",
            name="PATIENT DATA AGENT",
            role_description="Validates patient context, checks missing fields, and normalizes clinical parameters."
        )

    async def execute(self, agent_input: AgentInput) -> AgentOutput:
        p = agent_input.patient
        missing_fields: List[str] = []
        warnings: List[str] = []

        # Validate essential demographic & oncologic fields
        if not p.patient_id:
            missing_fields.append("patient_id")
        if not p.cancer_type:
            missing_fields.append("cancer_type")
        if not p.cancer_stage:
            missing_fields.append("cancer_stage")
        if p.age <= 0 or p.age > 120:
            warnings.append(f"Age {p.age} is outside customary clinical trial range.")

        # Laboratory checks
        if p.creatinine <= 0.1:
            warnings.append("Creatinine value suspiciously low; default reference assumed.")
        if p.ALT <= 0 or p.AST <= 0:
            warnings.append("Liver transaminases missing or zero.")

        # Normalization
        norm_cancer = p.cancer_type.strip()
        if "luad" in norm_cancer.lower() or "lung" in norm_cancer.lower():
            norm_cancer = "Lung (LUAD)"
        elif "brca" in norm_cancer.lower() or "breast" in norm_cancer.lower():
            norm_cancer = "Breast (BRCA)"
        elif "coad" in norm_cancer.lower() or "colon" in norm_cancer.lower():
            norm_cancer = "Colon (COAD)"

        norm_stage = p.cancer_stage.strip()
        if any(k in norm_stage.lower() for k in ["4", "iv"]):
            norm_stage = "Stage IV"
        elif any(k in norm_stage.lower() for k in ["3", "iii"]):
            norm_stage = "Stage III"
        elif any(k in norm_stage.lower() for k in ["2", "ii"]):
            norm_stage = "Stage II"
        elif any(k in norm_stage.lower() for k in ["1", "i"]):
            norm_stage = "Stage I"

        # Update patient context in place
        p.cancer_type = norm_cancer
        p.cancer_stage = norm_stage

        status = "Completed" if not missing_fields else "Warning"
        summary = (
            f"Validated patient {p.patient_id} ({norm_cancer}, {norm_stage}, age {p.age}). "
            f"Biomarker: {p.genomic_biomarker}, ctDNA: {p.ctDNA_level} ng/mL ({p.rising_ctdna_readings} rising readings). "
            f"Labs: Cr {p.creatinine} mg/dL, ALT {p.ALT} U/L, Bilirubin {p.bilirubin} mg/dL."
        )

        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.name,
            status=status,
            summary=summary,
            data={
                "normalized_patient": p.model_dump(),
                "missing_fields": missing_fields,
                "data_completeness_pct": round(100.0 - (len(missing_fields) * 15.0), 1)
            },
            confidence=0.98 if not missing_fields else 0.80,
            warnings=warnings
        )
