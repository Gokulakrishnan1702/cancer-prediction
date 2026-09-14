import os
import json
import logging
from fastapi import APIRouter, HTTPException
from api.schemas.audit import AuditReportResponse

logger = logging.getLogger("cdss_gateway")
router = APIRouter(prefix="/api/v1/reports", tags=["Reports & Audits"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
STRESS_TEST_REPORT_PATH = os.path.join(REPORTS_DIR, "model_stress_test_report.json")
DIVERGENCE_METRICS_PATH = os.path.join(REPORTS_DIR, "divergence_metrics.json")


@router.get(
    "/stress-test",
    response_model=AuditReportResponse,
    summary="Get model stress test and divergence audit",
    description="Serves the compiled stress-test evaluation report, divergence scores, and edge-case wildcard audit."
)
async def get_stress_test_report():
    if not os.path.exists(STRESS_TEST_REPORT_PATH):
        logger.warning(f"[REPORT NOT FOUND] Stress test report not found at {STRESS_TEST_REPORT_PATH}")
        raise HTTPException(
            status_code=404,
            detail={"error": "ReportNotFound", "message": "model_stress_test_report.json not found"}
        )

    try:
        with open(STRESS_TEST_REPORT_PATH, "r") as f:
            stress_data = json.load(f)

        divergence_data = None
        if os.path.exists(DIVERGENCE_METRICS_PATH):
            try:
                with open(DIVERGENCE_METRICS_PATH, "r") as f:
                    divergence_data = json.load(f)
            except Exception as e:
                logger.warning(f"[REPORT WARN] Could not load divergence metrics: {e}")

        payload = {
            "performance_decay": stress_data.get("performance_decay", {}),
            "cross_modal_safety": stress_data.get("cross_modal_safety", {}),
            "capstone_wildcard_audit": stress_data.get("capstone_wildcard_audit", {}),
            "wasserstein_kl_divergence": divergence_data
        }

        return AuditReportResponse.model_validate(payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[REPORT ERROR] Failed reading report: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "ReportReadError", "message": str(e)}
        )
