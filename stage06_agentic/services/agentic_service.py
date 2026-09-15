"""
Agentic Service: Singleton service coordinating the Multi-Agent engine,
audit logging, presets, and physician control operations.
"""
import logging
from typing import Dict, Any, List, Optional
from stage06_agentic.schemas.agent_schemas import PatientContext
from stage06_agentic.workflows.orchestrator import AgenticOrchestrator
from stage06_agentic.audit.audit_logger import get_audit_logger
from stage06_agentic.evaluation.test_cases import run_comprehensive_benchmark

logger = logging.getLogger("agentic_service")


class AgenticService:
    def __init__(self):
        self.orchestrator = AgenticOrchestrator()
        self.audit_logger = get_audit_logger()

    async def run_analysis(
        self,
        patient_data: Dict[str, Any],
        existing_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Runs the 10-agent deliberative oncology decision support workflow."""
        try:
            patient = PatientContext(**patient_data)
        except Exception as e:
            logger.warning(f"PatientContext parse warning: {e}. Attempting normalization.")
            patient = PatientContext()

        result = await self.orchestrator.execute_workflow(patient, existing_results)

        # Log into audit database
        try:
            self.audit_logger.log_workflow_execution(result)
        except Exception as e:
            logger.error(f"Audit log write failed: {e}")

        return result

    def pause_workflow(self) -> Dict[str, str]:
        self.orchestrator.pause()
        return {"status": "PAUSED", "message": "Agent execution paused by physician."}

    def resume_workflow(self) -> Dict[str, str]:
        self.orchestrator.resume()
        return {"status": "RESUMED", "message": "Agent execution resumed by physician."}

    def stop_workflow(self) -> Dict[str, str]:
        self.orchestrator.stop()
        return {"status": "STOPPED", "message": "Agent execution terminated by physician."}

    def apply_physician_override(
        self,
        trace_id: str,
        patient_id: str,
        reason: str,
        new_order: str = ""
    ) -> Dict[str, str]:
        self.orchestrator.override(reason)
        try:
            self.audit_logger.log_physician_override(
                trace_id=trace_id,
                patient_id=patient_id,
                reason=reason,
                new_order=new_order
            )
        except Exception as e:
            logger.error(f"Failed to record physician override: {e}")

        return {
            "status": "PHYSICIAN OVERRIDE ACTIVE",
            "message": f"Physician override logged: {reason}"
        }

    def get_presets(self) -> Dict[str, Any]:
        """Pre-curated clinical case studies from the slides and benchmark suite."""
        return {
            "case_a_egfr": {
                "name": "Case A: Patient A — EGFR+ NSCLC (Rapid Progression)",
                "description": "Severe risk, 6 rising ctDNA readings, 1 clinical trial slot available in SAVANNAH.",
                "data": {
                    "patient_id": "PAT-EGFR-001",
                    "age": 62.0,
                    "sex": "M",
                    "cancer_type": "Lung (LUAD)",
                    "cancer_stage": "Stage IV",
                    "genomic_biomarker": "EGFR L858R",
                    "mutation_profile": "EGFR L858R + TP53",
                    "mutation_count": 2,
                    "biomarker_status": "Positive",
                    "tumor_marker_level": 18.5,
                    "ctDNA_level": 0.88,
                    "rising_ctdna_readings": 6,
                    "treatment_name": "Osimertinib (Targeted TKI)",
                    "treatment_cycle": 4,
                    "dosage_mg": 80.0,
                    "previous_dose_mg": 80.0,
                    "treatment_duration_days": 84.0,
                    "treatment_history": "Prior Carboplatin + Pemetrexed; ongoing Osimertinib.",
                    "clinical_notes": "62yo male with metastatic EGFR+ NSCLC. Severe progression on serial imaging. 6 consecutive rising ctDNA timepoints indicating emergence of resistance bypass.",
                    "medical_image_provided": True,
                    "image_file_path": "/api/pipeline/image/lung/lung_001.png",
                    "creatinine": 1.15,
                    "bilirubin": 1.10,
                    "ALT": 48.0,
                    "AST": 44.0,
                    "platelet_count": 230.0,
                    "spo2": 97.0
                }
            },
            "case_b_kras": {
                "name": "Case B: Patient B — KRAS+ NSCLC (Stable Lesion)",
                "description": "Severe risk, 2 rising ctDNA readings, stable lesion, competing for limited slot.",
                "data": {
                    "patient_id": "PAT-KRAS-002",
                    "age": 59.0,
                    "sex": "F",
                    "cancer_type": "Lung (LUAD)",
                    "cancer_stage": "Stage IV",
                    "genomic_biomarker": "KRAS G12C",
                    "mutation_profile": "KRAS G12C",
                    "mutation_count": 1,
                    "biomarker_status": "Positive",
                    "tumor_marker_level": 12.0,
                    "ctDNA_level": 0.42,
                    "rising_ctdna_readings": 2,
                    "treatment_name": "Standard Chemotherapy",
                    "treatment_cycle": 3,
                    "dosage_mg": 500.0,
                    "previous_dose_mg": 500.0,
                    "treatment_duration_days": 63.0,
                    "treatment_history": "First-line Platinum-doublet chemotherapy.",
                    "clinical_notes": "59yo female with KRAS G12C NSCLC. Stable tumor size on follow-up CT, 2 rising ctDNA readings. Evaluating next-line therapy.",
                    "medical_image_provided": True,
                    "image_file_path": "/api/pipeline/image/lung/lung_002.png",
                    "creatinine": 1.05,
                    "bilirubin": 0.95,
                    "ALT": 36.0,
                    "AST": 34.0,
                    "platelet_count": 245.0,
                    "spo2": 98.0
                }
            },
            "case_c_dili": {
                "name": "Safety Edge Case: Severe Hepatic DILI / Hy's Law",
                "description": "ALT 280 U/L, Bilirubin 3.8 mg/dL — triggers SAFETY REVIEW REQUIRED halt.",
                "data": {
                    "patient_id": "PAT-DILI-003",
                    "age": 65.0,
                    "sex": "M",
                    "cancer_type": "Lung (LUAD)",
                    "cancer_stage": "Stage IV",
                    "genomic_biomarker": "EGFR L858R",
                    "mutation_profile": "EGFR L858R",
                    "mutation_count": 1,
                    "biomarker_status": "Positive",
                    "tumor_marker_level": 15.0,
                    "ctDNA_level": 0.65,
                    "rising_ctdna_readings": 3,
                    "treatment_name": "Targeted TKI",
                    "treatment_cycle": 2,
                    "dosage_mg": 80.0,
                    "previous_dose_mg": 80.0,
                    "treatment_duration_days": 42.0,
                    "treatment_history": "First-line Osimertinib.",
                    "clinical_notes": "Acute right upper quadrant pain, severe scleral icterus. ALT 280 U/L, Bilirubin 3.8 mg/dL.",
                    "medical_image_provided": True,
                    "image_file_path": "/api/pipeline/image/lung/lung_001.png",
                    "creatinine": 1.25,
                    "bilirubin": 3.8,  # Critical
                    "ALT": 280.0,      # Critical (>5x ULN)
                    "AST": 250.0,
                    "platelet_count": 190.0,
                    "spo2": 96.0
                }
            },
            "case_d_no_image": {
                "name": "Missing Data Edge Case: No Medical Image Provided",
                "description": "Pathology slides pending transfer — DL analysis safely unavailable.",
                "data": {
                    "patient_id": "PAT-NOIMG-004",
                    "age": 52.0,
                    "sex": "F",
                    "cancer_type": "Breast (BRCA)",
                    "cancer_stage": "Stage III",
                    "genomic_biomarker": "BRCA1",
                    "mutation_profile": "BRCA1 Deleterious",
                    "mutation_count": 1,
                    "biomarker_status": "Positive",
                    "tumor_marker_level": 14.0,
                    "ctDNA_level": 0.35,
                    "rising_ctdna_readings": 1,
                    "treatment_name": "Olaparib",
                    "treatment_cycle": 2,
                    "dosage_mg": 300.0,
                    "previous_dose_mg": 300.0,
                    "treatment_duration_days": 28.0,
                    "treatment_history": "Prior anthracycline chemotherapy.",
                    "clinical_notes": "Outpatient second opinion. Pathology slides and imaging pending transfer.",
                    "medical_image_provided": False,  # No image
                    "image_file_path": None,
                    "creatinine": 0.95,
                    "bilirubin": 0.8,
                    "ALT": 26.0,
                    "AST": 24.0,
                    "platelet_count": 220.0,
                    "spo2": 99.0
                }
            }
        }

    async def run_benchmark_evaluation(self) -> Dict[str, Any]:
        return await run_comprehensive_benchmark()


_agentic_service = None


def get_agentic_service() -> AgenticService:
    global _agentic_service
    if _agentic_service is None:
        _agentic_service = AgenticService()
    return _agentic_service
