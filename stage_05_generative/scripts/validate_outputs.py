import json
import os

def validate_outputs():
    print("[INFO] Starting Validation & Audit Suite...")
    
    input_path = os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs/synthetic_patient_vectors.json")
    output_path = os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs/synthetic_patient_vectors_validated.json")
    
    try:
        with open(input_path, "r") as f:
            patients = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read synthetic_patient_vectors.json: {e}")
        return
        
    valid_patients = []
    
    for p in patients:
        is_valid = True
        pid = p.get("synthetic_patient_id", "UNKNOWN")
        
        # 1. Schema Integrity
        required_keys = ["synthetic_patient_id", "baseline_genomics", "organ_impairment_baseline", "trajectory_t0_to_t4"]
        if not all(k in p for k in required_keys):
            print(f"[WARNING] {pid} failed schema integrity. Missing keys.")
            is_valid = False
            
        if is_valid and len(p["trajectory_t0_to_t4"]) != 5:
            print(f"[WARNING] {pid} has {len(p['trajectory_t0_to_t4'])} trajectory points instead of 5.")
            is_valid = False
            
        # 2. Medical Constraints
        if is_valid:
            for t in p["trajectory_t0_to_t4"]:
                vaf = t.get("ctdna_vaf")
                if vaf is None or not (0.0 <= vaf <= 1.0):
                    print(f"[WARNING] {pid} failed ctdna_vaf constraint: {vaf}")
                    is_valid = False
                    break
                    
                vol = t.get("tumor_vol_cm3")
                if vol is None or vol <= 0.0:
                    print(f"[WARNING] {pid} failed tumor_vol_cm3 constraint: {vol}")
                    is_valid = False
                    break
                    
        if is_valid:
            labs = p["organ_impairment_baseline"]
            for lab_key, valid_range in [("alt_u_l", (0, 2000)), ("ast_u_l", (0, 2000)), ("total_bilirubin_mg_dl", (0, 20))]:
                val = labs.get(lab_key)
                if val is not None:
                    if not (valid_range[0] <= val <= valid_range[1]):
                        print(f"[WARNING] {pid} failed clinical range for {lab_key}: {val}")
                        is_valid = False
                        break
                        
        if is_valid:
            valid_patients.append(p)
            
    with open(output_path, "w") as f:
        json.dump(valid_patients, f, indent=4)
        
    print(f"[SUCCESS] Validation complete. {len(valid_patients)}/{len(patients)} records passed. Saved to {output_path}")

if __name__ == "__main__":
    validate_outputs()
