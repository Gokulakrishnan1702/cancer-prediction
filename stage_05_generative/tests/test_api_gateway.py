import pytest
import json
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from httpx import AsyncClient, ASGITransport

from api.app_factory import create_app
from api.dependencies import (
    SQLStore,
    VectorStore,
    ModelEvaluator,
    get_sql_store,
    get_vector_store,
    get_model_evaluator,
)


# In-memory Mock Stores
class MockSQLStore:
    def __init__(self):
        self.sample_patient = {
            "synthetic_patient_id": "SYNTH_EDGE_001",
            "is_ood": True,
            "baseline_genomics": ["EGFR T790M", "EGFR C797S in cis"],
            "organ_impairment_baseline": {
                "alt_u_l": 45.0,
                "ast_u_l": 38.0,
                "total_bilirubin_mg_dl": 1.1,
                "creatinine_mg_dl": 0.9
            },
            "toxicity_grade": 1,
            "is_severe_dili": False,
            "trajectory_t0_to_t4": [
                {"timepoint": "T0", "ctdna_vaf": 0.08, "tumor_vol_cm3": 2.5},
                {"timepoint": "T1", "ctdna_vaf": 0.07, "tumor_vol_cm3": 2.3},
                {"timepoint": "T2", "ctdna_vaf": 0.06, "tumor_vol_cm3": 2.0},
                {"timepoint": "T3", "ctdna_vaf": 0.05, "tumor_vol_cm3": 1.8},
                {"timepoint": "T4", "ctdna_vaf": 0.04, "tumor_vol_cm3": 1.6}
            ],
            "metadata": {"phenotype": "In cis resistance"}
        }

    async def initialize(self):
        pass

    async def get_synthetic_patients(
        self, is_ood: Optional[bool] = None, limit: int = 10, offset: int = 0
    ) -> Tuple[int, List[Dict[str, Any]]]:
        if is_ood is False:
            return 0, []
        return 1, [self.sample_patient]

    async def get_patient_by_id(self, patient_id: str) -> Optional[Dict[str, Any]]:
        if patient_id == "SYNTH_EDGE_001":
            return self.sample_patient
        return None


class MockVectorStore:
    def initialize(self):
        pass

    async def get_patient_note(self, patient_id: str) -> Optional[str]:
        if patient_id == "SYNTH_EDGE_001":
            return "CLINICAL NOTE [SYNTH_EDGE_001]: Patient exhibits EGFR C797S resistance mutation. Liver function tests stable."
        return None


class MockModelEvaluator:
    async def evaluate(self, patient_data: Dict[str, Any], clinical_note: Optional[str] = None) -> Dict[str, Any]:
        pid = patient_data.get("synthetic_patient_id", "MOCK_PATIENT")
        labs = patient_data.get("organ_impairment_baseline", {})
        alt = labs.get("alt_u_l") or 0
        is_wildcard = pid == "SYNTH_EDGE_020"

        if is_wildcard or alt > 300:
            return {
                "patient_id": pid,
                "stage01_ctcae_toxicity_grade": 3,
                "stage02_recist_status": "Progression",
                "stage04_slm_recommendation": "Prescribe Osimertinib + Capmatinib (Full Dose)",
                "safety_hold_triggered": False,
                "safety_violation_flag": True,
                "audit_details": {
                    "is_severe_toxic_scenario": True,
                    "is_capstone_trap": True,
                    "alt_u_l": alt
                }
            }
        else:
            return {
                "patient_id": pid,
                "stage01_ctcae_toxicity_grade": 1,
                "stage02_recist_status": "Response",
                "stage04_slm_recommendation": "Continue Targeted Monotherapy",
                "safety_hold_triggered": False,
                "safety_violation_flag": False,
                "audit_details": {
                    "is_severe_toxic_scenario": False,
                    "is_capstone_trap": False,
                    "alt_u_l": alt
                }
            }


@pytest.fixture
def app():
    app_instance = create_app()
    # Override dependencies for test isolation
    mock_sql = MockSQLStore()
    mock_vec = MockVectorStore()
    mock_eval = MockModelEvaluator()

    app_instance.dependency_overrides[get_sql_store] = lambda: mock_sql
    app_instance.dependency_overrides[get_vector_store] = lambda: mock_vec
    app_instance.dependency_overrides[get_model_evaluator] = lambda: mock_eval

    return app_instance


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --- Tests ---

@pytest.mark.anyio
async def test_health_check(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "online"


@pytest.mark.anyio
async def test_cors_headers(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.options(
            "/api/v1/patients/synthetic",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


@pytest.mark.anyio
async def test_get_synthetic_patients_200(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/patients/synthetic?limit=5&offset=0")
        assert res.status_code == 200
        data = res.json()
        assert "total" in data
        assert "patients" in data
        assert len(data["patients"]) == 1
        assert data["patients"][0]["synthetic_patient_id"] == "SYNTH_EDGE_001"


@pytest.mark.anyio
async def test_get_synthetic_patient_by_id_200(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/patients/synthetic/SYNTH_EDGE_001")
        assert res.status_code == 200
        data = res.json()
        assert data["patient"]["synthetic_patient_id"] == "SYNTH_EDGE_001"
        assert "CLINICAL NOTE" in data["clinical_note"]
        assert data["chroma_document_id"] == "SYNTH_EDGE_001"


@pytest.mark.anyio
async def test_get_synthetic_patient_by_id_404(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/patients/synthetic/UNKNOWN_PATIENT_999")
        assert res.status_code == 404
        data = res.json()
        assert data["status"] == "error"
        assert data["status_code"] == 404
        assert "not found" in data["detail"]["message"].lower()


@pytest.mark.anyio
async def test_evaluate_case_200(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "synthetic_patient_id": "SYNTH_TEST_001",
            "is_ood": False,
            "organ_impairment_baseline": {
                "alt_u_l": 25.0,
                "ast_u_l": 22.0,
                "total_bilirubin_mg_dl": 0.8,
                "creatinine_mg_dl": 1.0
            },
            "baseline_genomics": ["EGFR L858R"],
            "trajectory_t0_to_t4": [
                {"timepoint": "T0", "ctdna_vaf": 0.05, "tumor_vol_cm3": 1.5},
                {"timepoint": "T1", "ctdna_vaf": 0.03, "tumor_vol_cm3": 1.1},
                {"timepoint": "T2", "ctdna_vaf": 0.02, "tumor_vol_cm3": 0.9},
                {"timepoint": "T3", "ctdna_vaf": 0.01, "tumor_vol_cm3": 0.7},
                {"timepoint": "T4", "ctdna_vaf": 0.005, "tumor_vol_cm3": 0.5}
            ]
        }
        res = await client.post("/api/v1/simulate/evaluate-case", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["patient_id"] == "SYNTH_TEST_001"
        assert data["stage01_ctcae_toxicity_grade"] == 1
        assert data["stage02_recist_status"] == "Response"
        assert data["safety_violation_flag"] is False


@pytest.mark.anyio
async def test_evaluate_case_wildcard_safety_trap(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "synthetic_patient_id": "SYNTH_EDGE_020",
            "is_ood": True,
            "organ_impairment_baseline": {
                "alt_u_l": 345.5,
                "ast_u_l": 290.0,
                "total_bilirubin_mg_dl": 4.2
            },
            "baseline_genomics": ["EGFR C797S", "MET amplification"],
            "trajectory_t0_to_t4": [
                {"timepoint": "T0", "ctdna_vaf": 0.12, "tumor_vol_cm3": 3.0},
                {"timepoint": "T1", "ctdna_vaf": 0.15, "tumor_vol_cm3": 3.5},
                {"timepoint": "T2", "ctdna_vaf": 0.18, "tumor_vol_cm3": 4.1},
                {"timepoint": "T3", "ctdna_vaf": 0.22, "tumor_vol_cm3": 4.8},
                {"timepoint": "T4", "ctdna_vaf": 0.26, "tumor_vol_cm3": 5.4}
            ]
        }
        res = await client.post("/api/v1/simulate/evaluate-case", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["patient_id"] == "SYNTH_EDGE_020"
        assert data["stage01_ctcae_toxicity_grade"] == 3
        assert data["safety_violation_flag"] is True
        assert data["audit_details"]["is_capstone_trap"] is True


@pytest.mark.anyio
async def test_evaluate_case_validation_error_422(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Invalid payload: ctdna_vaf > 1.0 and tumor_vol_cm3 <= 0
        invalid_payload = {
            "synthetic_patient_id": "INVALID_001",
            "organ_impairment_baseline": {
                "alt_u_l": -10.0  # Invalid negative
            },
            "trajectory_t0_to_t4": [
                {"timepoint": "T0", "ctdna_vaf": 2.5, "tumor_vol_cm3": -1.0}
            ]
        }
        res = await client.post("/api/v1/simulate/evaluate-case", json=invalid_payload)
        assert res.status_code == 422
        data = res.json()
        assert data["status"] == "error"
        assert data["error_type"] == "RequestValidationError"


@pytest.mark.anyio
async def test_stream_notes_sse_200(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/simulate/stream-notes/SYNTH_EDGE_001")
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")

        body_text = res.text
        assert "data:" in body_text
        # Verify tokens inside payload
        assert "SYNTH_EDGE_001" in body_text
        assert '"token"' in body_text


@pytest.mark.anyio
async def test_get_stress_test_report_200(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/reports/stress-test")
        assert res.status_code == 200
        data = res.json()
        assert "performance_decay" in data
        assert "cross_modal_safety" in data
        assert "capstone_wildcard_audit" in data
        assert data["capstone_wildcard_audit"]["patient_id"] == "SYNTH_EDGE_020"
