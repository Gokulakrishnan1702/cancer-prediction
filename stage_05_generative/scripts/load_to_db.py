import json
import os
import sqlite3
import pandas as pd

def load_to_db():
    print("[INFO] Starting DB Loader...")
    
    # 1. Ingest validated patients into SQLite
    validated_path = os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs/synthetic_patient_vectors_validated.json")
    db_path = os.path.join(os.path.dirname(__file__), "../data/stage05_eval.db")
    
    try:
        with open(validated_path, "r") as f:
            patients = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read validated vectors: {e}")
        return
        
    # Flatten patients for SQLite table
    flat_patients = []
    for p in patients:
        flat = {
            "synthetic_patient_id": p["synthetic_patient_id"],
            "baseline_genomics": ", ".join(p["baseline_genomics"]),
            "toxicity_grade": p.get("toxicity_grade", -1),
            "is_severe_dili": p.get("is_severe_dili", False),
            "alt_u_l": p["organ_impairment_baseline"].get("alt_u_l"),
            "ast_u_l": p["organ_impairment_baseline"].get("ast_u_l"),
            "total_bilirubin_mg_dl": p["organ_impairment_baseline"].get("total_bilirubin_mg_dl")
        }
        flat_patients.append(flat)
        
    df = pd.DataFrame(flat_patients)
    
    conn = sqlite3.connect(db_path)
    df.to_sql("synthetic_patients", conn, if_exists="replace", index=False)
    print(f"[SUCCESS] Loaded {len(df)} patient records into SQLite table 'synthetic_patients'")
    
    # 2. Ingest clinical notes into SQLite text index (since ChromaDB might not be installed, using SQLite FTS)
    notes_path = os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs/synthetic_clinical_notes.json")
    try:
        with open(notes_path, "r") as f:
            notes = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read clinical notes: {e}")
        conn.close()
        return
        
    df_notes = pd.DataFrame(notes)
    df_notes.to_sql("clinical_notes", conn, if_exists="replace", index=False)
    
    # Enable FTS for simple full text search as a mock for ChromaDB
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS clinical_notes_fts;")
    cursor.execute("CREATE VIRTUAL TABLE clinical_notes_fts USING fts5(synthetic_patient_id, clinical_note);")
    cursor.execute("INSERT INTO clinical_notes_fts SELECT synthetic_patient_id, clinical_note FROM clinical_notes;")
    conn.commit()
    
    print(f"[SUCCESS] Loaded {len(df_notes)} clinical notes into SQLite text index (FTS5)")
    conn.close()

if __name__ == "__main__":
    load_to_db()
