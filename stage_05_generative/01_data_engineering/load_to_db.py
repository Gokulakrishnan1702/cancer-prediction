import os
import json
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(DATA_DIR, "synthetic_outputs")
DB_PATH = os.path.join(DATA_DIR, "cdss_synthetic.db")


def load_to_db():
    """Ingests validated synthetic patient vectors and notes into SQLite database."""
    print("[DATA ENG DB] Starting database ingestion to cdss_synthetic.db...")

    validated_path = os.path.join(OUTPUTS_DIR, "synthetic_patient_vectors_validated.json")
    notes_path = os.path.join(OUTPUTS_DIR, "synthetic_clinical_notes.json")

    if not os.path.exists(validated_path):
        print(f"[DATA ENG ERROR] Validated patient file not found: {validated_path}")
        return

    with open(validated_path, "r") as f:
        patients = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS synthetic_patients (
            synthetic_patient_id TEXT PRIMARY KEY,
            is_ood INTEGER DEFAULT 0,
            baseline_genomics TEXT,
            toxicity_grade INTEGER,
            is_severe_dili INTEGER,
            alt_u_l REAL,
            ast_u_l REAL,
            total_bilirubin_mg_dl REAL,
            creatinine_mg_dl REAL,
            trajectory_json TEXT,
            raw_profile_json TEXT
        )
    """)

    for p in patients:
        pid = p["synthetic_patient_id"]
        labs = p.get("organ_impairment_baseline", {})
        genomics = ", ".join(p.get("baseline_genomics", []))
        tox = p.get("toxicity_grade", 0)
        dili = 1 if p.get("is_severe_dili") else 0
        is_ood = 1 if "EDGE" in pid or p.get("is_ood", False) else 0
        traj_json = json.dumps(p.get("trajectory_t0_to_t4", []))
        raw_json = json.dumps(p)

        cursor.execute("""
            INSERT OR REPLACE INTO synthetic_patients (
                synthetic_patient_id, is_ood, baseline_genomics,
                toxicity_grade, is_severe_dili, alt_u_l, ast_u_l,
                total_bilirubin_mg_dl, creatinine_mg_dl,
                trajectory_json, raw_profile_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pid, is_ood, genomics, tox, dili,
            labs.get("alt_u_l"), labs.get("ast_u_l"),
            labs.get("total_bilirubin_mg_dl"), labs.get("creatinine_mg_dl"),
            traj_json, raw_json
        ))

    conn.commit()
    print(f"[DATA ENG SUCCESS] Loaded {len(patients)} patient records into SQLite 'synthetic_patients'")

    # Load clinical notes if available
    if os.path.exists(notes_path):
        with open(notes_path, "r") as f:
            notes = json.load(f)
        df_notes = pd.DataFrame(notes)
        df_notes.to_sql("clinical_notes", conn, if_exists="replace", index=False)
        print(f"[DATA ENG SUCCESS] Loaded {len(df_notes)} notes into SQLite 'clinical_notes'")

    conn.close()


if __name__ == "__main__":
    load_to_db()
