"""
Stage Connectors: Read-only connectors to Stages 01–05
Accesses existing services without modifying any existing models or pipeline code.
"""
import os
import sys
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("agentic_stage_connectors")

# Ensure stage_05_generative is in sys.path for direct import
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGE5_DIR = os.path.join(WORKSPACE_ROOT, "stage_05_generative")
if STAGE5_DIR not in sys.path:
    sys.path.insert(0, STAGE5_DIR)


def get_pipeline_service():
    """Dynamically obtains the existing singleton UnifiedPipelineService."""
    try:
        from api.services.unified_pipeline_service import get_unified_pipeline_service
        return get_unified_pipeline_service()
    except Exception as e:
        logger.error(f"Could not import get_unified_pipeline_service: {e}")
        return None


async def call_stage1_ml(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Reads existing Stage 1 ML toxicity prediction (calibrated ensemble)."""
    service = get_pipeline_service()
    if service is None:
        return {"status": "Error", "message": "Unified pipeline service unavailable."}
    try:
        return service.run_stage1_ml(patient_dict)
    except Exception as e:
        logger.error(f"Error calling Stage 1 ML: {e}")
        return {"status": "Error", "message": str(e)}


async def call_stage2_dl(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Reads existing Stage 2 DL imaging prediction if image/scan provided."""
    service = get_pipeline_service()
    if service is None:
        return {"status": "Error", "message": "Unified pipeline service unavailable."}
    
    # Prompt rule: "If no image exists: 'DL analysis unavailable — image not provided.' Never generate fake image results."
    if not patient_dict.get("medical_image_provided", True):
        return {
            "stage": "DL",
            "status": "Unavailable",
            "image_prediction": "DL analysis unavailable — image not provided.",
            "classification": "None",
            "confidence": 0.0,
            "probability": 0.0,
            "risk_tier": "Unavailable",
            "clinical_interpretation": "DL analysis unavailable — image not provided."
        }

    try:
        return service.run_stage2_dl(patient_dict)
    except Exception as e:
        logger.error(f"Error calling Stage 2 DL: {e}")
        return {
            "stage": "DL",
            "status": "Unavailable",
            "image_prediction": "DL analysis unavailable — image not provided.",
            "error": str(e)
        }


async def call_stage3_nlp(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Reads existing Stage 3 NLP triage, entity extraction & urgency."""
    service = get_pipeline_service()
    if service is None:
        return {"status": "Error", "message": "Unified pipeline service unavailable."}
    try:
        return service.run_stage3_nlp(patient_dict)
    except Exception as e:
        logger.error(f"Error calling Stage 3 NLP: {e}")
        return {"status": "Error", "message": str(e)}


async def call_stage4_slm(
    patient_dict: Dict[str, Any],
    stage1_res: Optional[Dict[str, Any]] = None,
    stage2_res: Optional[Dict[str, Any]] = None,
    stage3_res: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Reads existing Stage 4 SLM concise physician briefing & guardrail."""
    service = get_pipeline_service()
    if service is None:
        return {"status": "Error", "message": "Unified pipeline service unavailable."}
    try:
        return service.run_stage4_slm(patient_dict, stage1_res, stage2_res, stage3_res)
    except Exception as e:
        logger.error(f"Error calling Stage 4 SLM: {e}")
        return {"status": "Error", "message": str(e)}


async def call_stage5_genai(
    patient_dict: Dict[str, Any],
    stage1_res: Optional[Dict[str, Any]] = None,
    stage2_res: Optional[Dict[str, Any]] = None,
    stage3_res: Optional[Dict[str, Any]] = None,
    stage4_res: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Reads existing Stage 5 GenAI clinical stress-test & simulated scenarios."""
    service = get_pipeline_service()
    if service is None:
        return {"status": "Error", "message": "Unified pipeline service unavailable."}
    try:
        return service.run_stage5_genai(patient_dict, stage1_res, stage2_res, stage3_res, stage4_res)
    except Exception as e:
        logger.error(f"Error calling Stage 5 GenAI: {e}")
        return {"status": "Error", "message": str(e)}
