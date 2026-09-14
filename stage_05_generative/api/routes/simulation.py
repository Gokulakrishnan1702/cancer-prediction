import json
import asyncio
import logging
from typing import Optional, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.schemas.patient import SyntheticPatientProfile
from api.schemas.evaluation import ModelEvaluationResponse
from api.dependencies import (
    ModelEvaluator,
    VectorStore,
    SQLStore,
    get_model_evaluator,
    get_vector_store,
    get_sql_store,
)

logger = logging.getLogger("cdss_gateway")
router = APIRouter(prefix="/api/v1/simulate", tags=["Simulation & Inference"])


@router.post(
    "/evaluate-case",
    response_model=ModelEvaluationResponse,
    summary="Evaluate synthetic patient case",
    description="Runs live cross-modal inference across Stage 01 (Toxicity), Stage 02 (Progression), and Stage 04 (SLM Decision)."
)
async def evaluate_case(
    patient_profile: SyntheticPatientProfile,
    model_evaluator: ModelEvaluator = Depends(get_model_evaluator),
    vector_store: VectorStore = Depends(get_vector_store),
):
    try:
        logger.info(f"[EVAL SIMULATION] Received simulation request for patient {patient_profile.synthetic_patient_id}")
        patient_dict = patient_profile.model_dump()

        # Check for clinical note in vector store
        clinical_note = await vector_store.get_patient_note(patient_profile.synthetic_patient_id)

        # Run live multi-stage inference
        evaluation_result = await model_evaluator.evaluate(patient_dict, clinical_note)
        logger.info(
            f"[EVAL SIMULATION] Evaluated {patient_profile.synthetic_patient_id}: "
            f"Stage01={evaluation_result['stage01_ctcae_toxicity_grade']}, "
            f"Stage02={evaluation_result['stage02_recist_status']}, "
            f"Violation={evaluation_result['safety_violation_flag']}"
        )

        return ModelEvaluationResponse.model_validate(evaluation_result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[EVAL SIMULATION ERROR] Inference pipeline failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "InferencePipelineError", "message": str(e)}
        )


@router.get(
    "/stream-notes/{patient_id}",
    summary="Stream SLM clinical progress notes via SSE",
    description="Streams generated SLM clinical progress notes token-by-token using Server-Sent Events (SSE)."
)
async def stream_clinical_notes(
    patient_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
    sql_store: SQLStore = Depends(get_sql_store),
):
    logger.info(f"[SSE STREAMING] Initiating real-time note stream for patient {patient_id}")

    # Retrieve note from ChromaDB or reconstruct from profile
    note = await vector_store.get_patient_note(patient_id)
    if not note:
        patient_data = await sql_store.get_patient_by_id(patient_id)
        if patient_data:
            alt = patient_data.get("organ_impairment_baseline", {}).get("alt_u_l", 35)
            genomics = ", ".join(patient_data.get("baseline_genomics", ["EGFR L858R"]))
            note = (
                f"PATIENT ASSESSMENT & CLINICAL NOTE [{patient_id}]: "
                f"Patient presents with baseline genomics ({genomics}). "
                f"Current hepatic biomarker monitoring reflects ALT at {alt} U/L. "
                f"Evaluating targeted kinase inhibitor tolerance, safety holds, and treatment trajectory."
            )
        else:
            note = (
                f"PATIENT ASSESSMENT & CLINICAL NOTE [{patient_id}]: "
                f"Synthetic profile initialized. Out-of-distribution biomarker stress simulation underway. "
                f"Monitoring for severe Drug-Induced Liver Injury (DILI) and emergence of resistance mutations."
            )

    async def event_generator() -> AsyncGenerator[dict, None]:
        # Split into words/tokens for streaming
        tokens = note.split(" ")
        for i, token in enumerate(tokens):
            payload = {
                "patient_id": patient_id,
                "token": token + (" " if i < len(tokens) - 1 else ""),
                "index": i,
                "done": i == len(tokens) - 1
            }
            yield {
                "event": "message",
                "data": json.dumps(payload)
            }
            # Simulating realistic SLM generation latency (20ms per token)
            await asyncio.sleep(0.02)
        logger.info(f"[SSE STREAMING] Completed note streaming for patient {patient_id} ({len(tokens)} tokens)")

    return EventSourceResponse(event_generator())
