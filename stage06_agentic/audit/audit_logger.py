"""
Audit Logger for Stage 06 Agentic AI
Maintains persistent records of:
- Agent execution traces
- Physician overrides ("PHYSICIAN OVERRIDE ACTIVE")
- Safety violations and workflow halts
"""
import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(AUDIT_DIR, "stage06_audit.db")


class AuditLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_audits (
                    trace_id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    timestamp TEXT,
                    status TEXT,
                    execution_time_ms REAL,
                    safety_status TEXT,
                    physician_override_active INTEGER,
                    override_notes TEXT,
                    decision_traces TEXT,
                    final_recommendation TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS physician_overrides (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    patient_id TEXT,
                    timestamp TEXT,
                    physician_action TEXT,
                    override_reason TEXT,
                    original_recommendation TEXT,
                    new_clinical_order TEXT
                )
            """)
            conn.commit()

    def log_workflow_execution(self, workflow_result: Dict[str, Any]) -> str:
        final_assessment = workflow_result.get("final_assessment", {}) or {}
        trace_id = final_assessment.get("audit_trace_id", f"TRACE-{int(datetime.now().timestamp())}")
        pid = workflow_result.get("patient_id", "UNKNOWN")
        ts = datetime.now(timezone.utc).isoformat()
        status = workflow_result.get("status", "SUCCESS")
        exec_ms = workflow_result.get("execution_time_ms", 0.0)
        safety_status = workflow_result.get("safety_status", "PASSED")
        override_active = 1 if workflow_result.get("physician_override_active") else 0
        override_notes = workflow_result.get("physician_override_notes", "")
        traces_json = json.dumps(workflow_result.get("decision_traces", []))
        rec = final_assessment.get("final_recommendation", "")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO workflow_audits (
                    trace_id, patient_id, timestamp, status, execution_time_ms,
                    safety_status, physician_override_active, override_notes,
                    decision_traces, final_recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace_id, pid, ts, status, exec_ms, safety_status,
                override_active, override_notes, traces_json, rec
            ))
            conn.commit()

        return trace_id

    def log_physician_override(
        self,
        trace_id: str,
        patient_id: str,
        reason: str,
        original_rec: str = "",
        new_order: str = ""
    ):
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO physician_overrides (
                    trace_id, patient_id, timestamp, physician_action,
                    override_reason, original_recommendation, new_clinical_order
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (trace_id, patient_id, ts, "PHYSICIAN OVERRIDE ACTIVE", reason, original_rec, new_order))

            # Also update audit table
            cursor.execute("""
                UPDATE workflow_audits
                SET physician_override_active = 1, override_notes = ?, status = 'OVERRIDDEN'
                WHERE trace_id = ?
            """, (reason, trace_id))
            conn.commit()

    def get_recent_audits(self, limit: int = 20) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM workflow_audits ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_override_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM physician_overrides ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]


_singleton_logger = None


def get_audit_logger() -> AuditLogger:
    global _singleton_logger
    if _singleton_logger is None:
        _singleton_logger = AuditLogger()
    return _singleton_logger
