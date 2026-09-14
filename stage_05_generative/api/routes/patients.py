import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from api.schemas.patient import (
    SyntheticPatientProfile,
    PatientPaginationResponse,
    MultiModalPatientResponse,
)
from api.dependencies import SQLStore, VectorStore, get_sql_store, get_vector_store

logger = logging.getLogger("cdss_gateway")
router = APIRouter(prefix="/api/v1/patients", tags=["Synthetic Patients"])


@router.get(
    "/synthetic",
    response_model=PatientPaginationResponse,
    summary="List synthetic patient profiles",
    description="Fetches paginated synthetic patient vectors with optional Out-of-Distribution (OOD) filter."
)
async def list_synthetic_patients(
    is_ood: Optional[bool] = Query(None, description="Filter by Out-of-Distribution status"),
    limit: int = Query(10, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    sql_store: SQLStore = Depends(get_sql_store),
):
    try:
        total, raw_patients = await sql_store.get_synthetic_patients(
            is_ood=is_ood, limit=limit, offset=offset
        )
        patient_profiles = [SyntheticPatientProfile.model_validate(p) for p in raw_patients]

        return PatientPaginationResponse(
            total=total,
            limit=limit,
            offset=offset,
            patients=patient_profiles,
        )
    except Exception as e:
        logger.error(f"[PATIENT QUERY ERROR] Failed to query synthetic patients: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "DatabaseQueryError", "message": str(e)}
        )


@router.get(
    "/synthetic/{patient_id}",
    response_model=MultiModalPatientResponse,
    summary="Get multi-modal patient details",
    description="Retrieves full multi-modal payload (structured labs + unstructured EHR clinical notes from ChromaDB)."
)
async def get_synthetic_patient_by_id(
    patient_id: str,
    sql_store: SQLStore = Depends(get_sql_store),
    vector_store: VectorStore = Depends(get_vector_store),
):
    patient_data = await sql_store.get_patient_by_id(patient_id)
    if not patient_data:
        logger.warning(f"[PATIENT NOT FOUND] Patient {patient_id} does not exist in SQLStore")
        raise HTTPException(
            status_code=404,
            detail={"error": "PatientNotFound", "message": f"Synthetic patient '{patient_id}' not found"}
        )

    # Fetch unstructured note from ChromaDB
    clinical_note = await vector_store.get_patient_note(patient_id)
    if not clinical_note:
        # Fallback check in raw profile if embedded
        clinical_note = patient_data.get("clinical_note")

    profile = SyntheticPatientProfile.model_validate(patient_data)
    return MultiModalPatientResponse(
        patient=profile,
        clinical_note=clinical_note,
        chroma_document_id=patient_id if clinical_note else None
    )
