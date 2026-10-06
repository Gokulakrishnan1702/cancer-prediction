"""
FastAPI Routes for Unified 5-Stage Clinical AI Pipeline
======================================================
Provides endpoints for:
  - Full sequential 5-stage pipeline execution
  - Individual stage execution (ML, DL, NLP, SLM, GenAI)
  - Pre-curated patient presets
  - CSV upload and parsing
  - Historical patient analyses
  - Hospital analytics and CDSS performance metrics
  - Medical scan and pathology image streaming
"""

import os
import io
import csv
import glob
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.services.unified_pipeline_service import get_unified_pipeline_service

logger = logging.getLogger("pipeline_routes")
router = APIRouter(prefix="/api/pipeline", tags=["Unified Clinical AI Pipeline"])

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STAGE2_IMG_DIR = os.path.join(WORKSPACE_ROOT, "Stage2", "2d_images_large")


class PatientAnalysisRequest(BaseModel):
    patient_id: Optional[str] = "PAT-0001"
    age: Optional[float] = 60.0
    sex: Optional[str] = "M"
    cancer_type: Optional[str] = "Lung (LUAD)"
    cancer_stage: Optional[str] = "Stage IV"
    treatment_name: Optional[str] = "Targeted Therapy"
    treatment_cycle: Optional[int] = 3
    dosage_mg: Optional[float] = 400.0
    previous_dose_mg: Optional[float] = 400.0
    dose_change_percentage: Optional[float] = 0.0
    treatment_duration_days: Optional[float] = 21.0
    mutation_profile: Optional[str] = "EGFR L858R"
    mutation_count: Optional[int] = 1
    biomarker_status: Optional[str] = "Positive"
    ctDNA_level: Optional[float] = 0.45
    tumor_marker_level: Optional[float] = 14.2
    WBC: Optional[float] = 6.5
    hemoglobin: Optional[float] = 12.5
    platelet_count: Optional[float] = 250.0
    creatinine: Optional[float] = 1.1
    bilirubin: Optional[float] = 1.2
    ALT: Optional[float] = 45.0
    AST: Optional[float] = 42.0
    heart_rate: Optional[float] = 76.0
    systolic_bp: Optional[float] = 130.0
    diastolic_bp: Optional[float] = 82.0
    spo2: Optional[float] = 96.0
    temperature: Optional[float] = 37.0
    respiratory_rate: Optional[float] = 16.0
    previous_adverse_reaction: Optional[str] = "No"
    previous_treatment_response: Optional[str] = "Partial Response"
    previous_toxicity: Optional[str] = "Low"
    treatment_response: Optional[str] = "Partial Response"
    clinical_notes: Optional[str] = None
    image_file: Optional[str] = None


@router.post("/run-all", summary="Run Complete 6-Stage AI Analysis")
async def run_complete_analysis(req: PatientAnalysisRequest):
    """Executes ML -> DL -> NLP -> SLM -> GenAI -> Agentic sequentially and returns unified report."""
    try:
        service = get_unified_pipeline_service()
        patient_dict = req.model_dump()
        result = service.run_full_pipeline(patient_dict, image_filename=req.image_file)
        return result
    except Exception as e:
        logger.error(f"[RUN-ALL ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stage/{stage_id}", summary="Run Individual AI Stage")
async def run_single_stage(stage_id: str, req: PatientAnalysisRequest):
    """Runs a single isolated stage (1=ML, 2=DL, 3=NLP, 4=SLM, 5=GenAI, 6=Agentic)."""
    service = get_unified_pipeline_service()
    patient_dict = req.model_dump()
    st = stage_id.lower()

    if st in ["1", "ml", "stage1"]:
        return service.run_stage1_ml(patient_dict)
    elif st in ["2", "dl", "stage2"]:
        return service.run_stage2_dl(patient_dict, image_filename=req.image_file)
    elif st in ["3", "nlp", "stage3"]:
        return service.run_stage3_nlp(patient_dict)
    elif st in ["4", "slm", "stage4"]:
        return service.run_stage4_slm(patient_dict)
    elif st in ["5", "genai", "stage5"]:
        return service.run_stage5_genai(patient_dict)
    elif st in ["6", "agentic", "stage6"]:
        return service.run_stage6_agentic(patient_dict)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown stage '{stage_id}'. Valid values: 1, 2, 3, 4, 5, 6.")


@router.get("/presets", summary="Get Pre-configured Clinical Patient Presets")
async def get_patient_presets():
    """Returns realistic clinical cases (High Risk, Moderate Risk, Low Risk, Critical DILI Edge Case)."""
    service = get_unified_pipeline_service()
    return service.get_sample_presets()


@router.get("/history", summary="Get Patient Analysis History")
async def get_patient_history(limit: int = 50):
    """Fetches previously analyzed patient cases from SQLite database."""
    service = get_unified_pipeline_service()
    return await service.get_history(limit=limit)


@router.get("/analytics", summary="Get CDSS Hospital Analytics & Model Metrics")
async def get_cdss_analytics():
    """Serves high-level hospital decision-support performance analytics and charts."""
    service = get_unified_pipeline_service()
    return await service.get_analytics()


@router.post("/upload-csv", summary="Upload and Parse Patient Data CSV")
async def upload_patient_csv(file: UploadFile = File(...)):
    """Parses a patient/lab CSV file and maps fields into structured clinical input schema."""
    try:
        contents = await file.read()
        text_stream = io.StringIO(contents.decode("utf-8", errors="ignore"))
        reader = csv.DictReader(text_stream)
        rows = list(reader)

        if not rows:
            raise HTTPException(status_code=400, detail="CSV file is empty or has invalid formatting.")

        first_row = rows[0]

        # Case-insensitive mapping helper
        def get_field(keys, default=None):
            for k in keys:
                for row_k, v in first_row.items():
                    if row_k.strip().lower() == k.lower() and v != "":
                        return v
            return default

        mapped_data = {
            "patient_id": get_field(["patient_id", "id", "mrn"], "PAT-UPLOAD-01"),
            "age": float(get_field(["age"], 62.0)),
            "sex": get_field(["sex", "gender"], "M"),
            "cancer_type": get_field(["cancer_type", "cancer", "histology"], "Lung (LUAD)"),
            "cancer_stage": get_field(["cancer_stage", "stage", "stage_numeric"], "Stage IV"),
            "treatment_name": get_field(["treatment_name", "treatment_type", "drug"], "Targeted Therapy"),
            "treatment_cycle": int(float(get_field(["treatment_cycle", "cycle"], 3))),
            "dosage_mg": float(get_field(["dosage_mg", "dose"], 400.0)),
            "previous_dose_mg": float(get_field(["previous_dose_mg"], 400.0)),
            "dose_change_percentage": float(get_field(["dose_change_percentage"], 0.0)),
            "treatment_duration_days": float(get_field(["treatment_duration_days", "duration"], 21.0)),
            "mutation_profile": get_field(["mutation_profile", "genomics", "mutations"], "EGFR L858R"),
            "mutation_count": int(float(get_field(["mutation_count"], 1))),
            "biomarker_status": get_field(["biomarker_status"], "Positive"),
            "ctDNA_level": float(get_field(["ctDNA_level", "baseline_cdna", "ctdna"], 0.45)),
            "tumor_marker_level": float(get_field(["tumor_marker_level", "baseline_cea"], 14.2)),
            "WBC": float(get_field(["WBC", "baseline_wbc"], 6.5)),
            "hemoglobin": float(get_field(["hemoglobin", "baseline_hgb"], 12.5)),
            "platelet_count": float(get_field(["platelet_count", "baseline_plt"], 250.0)),
            "creatinine": float(get_field(["creatinine", "baseline_creatinine"], 1.1)),
            "bilirubin": float(get_field(["bilirubin", "baseline_bilirubin"], 1.2)),
            "ALT": float(get_field(["ALT", "baseline_alt"], 45.0)),
            "AST": float(get_field(["AST", "baseline_ast"], 42.0)),
            "heart_rate": float(get_field(["heart_rate"], 76.0)),
            "systolic_bp": float(get_field(["systolic_bp"], 130.0)),
            "diastolic_bp": float(get_field(["diastolic_bp"], 82.0)),
            "spo2": float(get_field(["spo2"], 96.0)),
            "temperature": float(get_field(["temperature"], 37.0)),
            "respiratory_rate": float(get_field(["respiratory_rate"], 16.0)),
            "clinical_notes": get_field(["clinical_notes", "notes", "symptoms"], "Uploaded via CSV panel.")
        }

        return {
            "status": "success",
            "message": f"Successfully parsed CSV with {len(rows)} record(s). Loaded patient {mapped_data['patient_id']}.",
            "patient_data": mapped_data,
            "total_records_found": len(rows)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CSV UPLOAD ERROR] {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")


@router.get("/sample-images", summary="List Sample Medical Images")
async def list_sample_images():
    """Lists available CT slices and pathology tiles grouped by cancer type."""
    results = {}
    if os.path.exists(STAGE2_IMG_DIR):
        for ctype in os.listdir(STAGE2_IMG_DIR):
            cpath = os.path.join(STAGE2_IMG_DIR, ctype)
            if os.path.isdir(cpath):
                files = os.listdir(cpath)
                cts = [f for f in files if "ct_slice" in f][:5]
                tiles = [f for f in files if "tile" in f][:5]
                results[ctype] = {
                    "ct_scans": cts,
                    "pathology_tiles": tiles
                }
    return results


@router.get("/image/{cancer_type}/{image_name}", summary="Serve Medical Image Scan / Tile")
async def serve_medical_image(cancer_type: str, image_name: str):
    """Serves real CT scan slices and Pathology tiles for the DL visualizer."""
    safe_ctype = os.path.basename(cancer_type).lower()
    safe_img = os.path.basename(image_name)
    target_path = os.path.join(STAGE2_IMG_DIR, safe_ctype, safe_img)

    if not os.path.exists(target_path):
        fallback = os.path.join(STAGE2_IMG_DIR, "luad", "ct_slice_0000.png")
        if os.path.exists(fallback):
            return FileResponse(fallback, media_type="image/png")
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(target_path, media_type="image/png")


class PatientPdfExportRequest(BaseModel):
    patient: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None


@router.post("/export-pdf", summary="Export Clinical Tumor Board PDF Report")
async def export_patient_pdf(req: PatientPdfExportRequest):
    """Generates an oncologist-grade, downloadable PDF report for the active patient case."""
    try:
        from api.services.pdf_report_service import generate_patient_pdf_report
        pdf_bytes = generate_patient_pdf_report(req.patient, req.results)
        pid = req.patient.get("patient_id", "PAT-0001")
        filename = f"TumorBoard_Report_{pid}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        logger.error(f"[PDF EXPORT ERROR] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate clinical PDF: {str(e)}")

