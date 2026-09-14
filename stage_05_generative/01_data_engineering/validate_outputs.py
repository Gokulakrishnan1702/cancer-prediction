import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(DATA_DIR, "synthetic_outputs")


def validate_synthetic_outputs():
    """Validates schema integrity and physiological boundaries on synthetic patient profiles."""
    print("[DATA ENG VALIDATE] Starting Schema & Medical Constraints Validation...")

    input_path = os.path.join(OUTPUTS_DIR, "synthetic_patient_vectors.json")
    output_path = os.path.join(OUTPUTS_DIR, "synthetic_patient_vectors_validated.json")

    if not os.path.exists(input_path):
        print(f"[DATA ENG ERROR] Input file not found: {input_path}")
        return []

    with open(input_path, "r") as f:
        patients = json.load(f)

    valid_patients = []
    for p in patients:
        is_valid = True
        pid = p.get("synthetic_patient_id", "UNKNOWN")

        # 1. Schema Integrity
        required_keys = ["synthetic_patient_id", "baseline_genomics", "organ_impairment_baseline", "trajectory_t0_to_t4"]
        if not all(k in p for k in required_keys):
            print(f"[DATA ENG WARNING] {pid} failed schema integrity: Missing required keys.")
            is_valid = False

        if is_valid and len(p["trajectory_t0_to_t4"]) != 5:
            print(f"[DATA ENG WARNING] {pid} has {len(p['trajectory_t0_to_t4'])} trajectory points (must be 5).")
            is_valid = False

        # 2. Medical Bounds
        if is_valid:
            for t in p["trajectory_t0_to_t4"]:
                vaf = t.get("ctdna_vaf")
                if vaf is None or not (0.0 <= vaf <= 1.0):
                    print(f"[DATA ENG WARNING] {pid} failed ctdna_vaf constraint: {vaf}")
                    is_valid = False
                    break

                vol = t.get("tumor_vol_cm3")
                if vol is None or vol <= 0.0:
                    print(f"[DATA ENG WARNING] {pid} failed tumor_vol_cm3 constraint: {vol}")
                    is_valid = False
                    break

        if is_valid:
            labs = p["organ_impairment_baseline"]
            bounds = [("alt_u_l", (0, 2000)), ("ast_u_l", (0, 2000)), ("total_bilirubin_mg_dl", (0, 20))]
            for lab_key, valid_range in bounds:
                val = labs.get(lab_key)
                if val is not None and not (valid_range[0] <= val <= valid_range[1]):
                    print(f"[DATA ENG WARNING] {pid} failed clinical range for {lab_key}: {val}")
                    is_valid = False
                    break

        if is_valid:
            valid_patients.append(p)

    with open(output_path, "w") as f:
        json.dump(valid_patients, f, indent=4)

    print(f"[DATA ENG SUCCESS] Validation complete: {len(valid_patients)}/{len(patients)} records passed -> {output_path}")
    return valid_patients


if __name__ == "__main__":
    validate_synthetic_outputs()
