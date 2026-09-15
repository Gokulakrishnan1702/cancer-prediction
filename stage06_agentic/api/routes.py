"""
FastAPI Routes for Stage 06 Agentic AI
Provides REST endpoints for Multi-Agent Oncology Decision Engine
"""
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from stage06_agentic.services.agentic_service import get_agentic_service

logger = logging.getLogger("agentic_api")
router = APIRouter(prefix="/api/v1/agentic", tags=["Stage 06 — Agentic AI"])


class OverrideRequest(BaseModel):
    trace_id: str = "TRACE-001"
    patient_id: str = "PAT-0001"
    reason: str = "Physician clinical decision override"
    new_order: Optional[str] = "Maintain standard-of-care systemic therapy."


@router.get("/health")
async def health_check():
    return {
        "status": "online",
        "stage": "Stage 06 — Agentic AI",
        "description": "Autonomous Multi-Agent Oncology Decision Engine"
    }


@router.get("/presets")
async def get_presets():
    service = get_agentic_service()
    return service.get_presets()


@router.post("/run")
async def run_agentic_workflow(payload: Dict[str, Any] = Body(...)):
    service = get_agentic_service()
    patient_data = payload.get("patient", payload)
    existing_results = payload.get("existing_results")
    result = await service.run_analysis(patient_data, existing_results)
    return result


@router.post("/pause")
async def pause_agent():
    service = get_agentic_service()
    return service.pause_workflow()


@router.post("/resume")
async def resume_agent():
    service = get_agentic_service()
    return service.resume_workflow()


@router.post("/stop")
async def stop_agent():
    service = get_agentic_service()
    return service.stop_workflow()


@router.post("/override")
async def physician_override(req: OverrideRequest):
    service = get_agentic_service()
    return service.apply_physician_override(
        trace_id=req.trace_id,
        patient_id=req.patient_id,
        reason=req.reason,
        new_order=req.new_order or ""
    )


@router.get("/trials")
async def get_clinical_trials():
    from stage06_agentic.knowledge import load_clinical_trials
    return load_clinical_trials()


@router.get("/audits")
async def get_audit_trail():
    service = get_agentic_service()
    return {
        "audits": service.audit_logger.get_recent_audits(limit=25),
        "overrides": service.audit_logger.get_override_logs(limit=25)
    }


@router.post("/evaluate")
async def run_evaluation_benchmark():
    service = get_agentic_service()
    metrics = await service.run_benchmark_evaluation()
    return metrics
