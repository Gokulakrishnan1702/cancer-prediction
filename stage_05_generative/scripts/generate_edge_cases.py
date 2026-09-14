import json
import os
import random
import argparse
import copy

def generate_edge_cases(num_cases, seed, out_dir):
    print(f"[INFO] Starting Synthetic Edge Case Generator (Seed: {seed}, Cases: {num_cases})...")
    random.seed(seed)
    
    input_path = os.path.join(os.path.dirname(__file__), "../data/process_seeds/baseline_dists.json")
    try:
        with open(input_path, "r") as f:
            baselines = json.load(f)
        print("[SUCCESS] Loaded baseline_dists.json")
    except Exception as e:
        print(f"[ERROR] Could not load baselines: {e}")
        return
        
    patients = []
    notes = []
    
    dili_count = int(num_cases * 0.3)
    dili_indices = set(random.sample(range(num_cases - 1), dili_count)) if num_cases > 1 else set()
    dili_indices.add(num_cases - 1) # Capstone Wildcard Dual-Driver Hepatotoxic Trap at last index
    
    for i in range(num_cases):
        patient_id = f"SYNTH_EDGE_{i+1:03d}"
        
        # Genomics
        mutations = []
        if i == num_cases - 1:
            mutations = ["EGFR C797S", "MET amplification"]
        elif i % 3 == 0:
            mutations = ["EGFR T790M", "EGFR C797S in cis"]
        elif i % 3 == 1:
            mutations = ["KRAS G12C", "MET amplification"]
        else:
            mutations = ["ALK rearrangement", "TP53"]
            
        # Organ Impairment (DILI)
        is_dili = i in dili_indices
        if is_dili:
            alt = round(random.uniform(300.0, 500.0), 1)
            ast = round(random.uniform(300.0, 500.0), 1)
            bilirubin = round(random.uniform(3.0, 6.5), 1)
            tox_grade = 3
        else:
            alt = round(random.uniform(10.0, 40.0), 1)
            ast = round(random.uniform(10.0, 40.0), 1)
            bilirubin = round(random.uniform(0.3, 1.2), 1)
            try:
                tox_grade = int(random.choice(list(baselines["toxicity"]["grade_probabilities"].keys())))
            except:
                tox_grade = 1
            
        # Trajectories
        trajectory = []
        ctdna_base = random.uniform(0.001, 0.1) # [0.0, 1.0] bound
        tumor_base = random.uniform(0.02, 0.1) # strictly > 0.0 cm3
        
        progression_factor = 1.2 if "MET amplification" in mutations or "EGFR C797S" in mutations else 0.8
        
        for t in range(5):
            ctdna = min(round(ctdna_base * (progression_factor ** t) * random.uniform(0.9, 1.1), 4), 1.0)
            tumor = round(tumor_base * (progression_factor ** t) * random.uniform(0.95, 1.05), 3)
            trajectory.append({
                "timepoint": f"T{t}",
                "ctdna_vaf": ctdna,
                "tumor_vol_cm3": tumor
            })
            
        patient_data = {
            "synthetic_patient_id": patient_id,
            "baseline_genomics": mutations,
            "organ_impairment_baseline": {
                "alt_u_l": alt,
                "ast_u_l": ast,
                "total_bilirubin_mg_dl": bilirubin
            },
            "toxicity_grade": tox_grade,
            "is_severe_dili": is_dili,
            "trajectory_t0_to_t4": trajectory
        }
        
        # Null Injection
        if random.random() < 0.1:
            patient_data["organ_impairment_baseline"]["alt_u_l"] = None
        if random.random() < 0.1:
            patient_data["organ_impairment_baseline"]["ast_u_l"] = None
            
        patients.append(patient_data)
        
        # Clinical Note
        note_text = f"Patient {patient_id} presents with {', '.join(mutations)}."
        if is_dili:
            note_text += f" Labs show severe hepatotoxicity (Grade 3 DILI) with ALT {alt} U/L and Total Bilirubin {bilirubin} mg/dL."
        else:
            note_text += f" Labs are within normal limits (ALT {alt} U/L, Tbili {bilirubin} mg/dL)."
            
        note_text += f" Tumor volume progressed from {trajectory[0]['tumor_vol_cm3']} cm3 at T0 to {trajectory[4]['tumor_vol_cm3']} cm3 at T4."
        
        notes.append({
            "synthetic_patient_id": patient_id,
            "clinical_note": note_text
        })
        
    # Near-Duplicate Injection (SYNTH_EDGE_021)
    if len(patients) > 0:
        clone = copy.deepcopy(patients[0])
        clone["synthetic_patient_id"] = "SYNTH_EDGE_021"
        for point in clone["trajectory_t0_to_t4"]:
            point["tumor_vol_cm3"] = round(point["tumor_vol_cm3"] + 0.01, 3)
        patients.append(clone)
        
        clone_note_text = notes[0]["clinical_note"].replace("SYNTH_EDGE_001", "SYNTH_EDGE_021")
        notes.append({
            "synthetic_patient_id": "SYNTH_EDGE_021",
            "clinical_note": clone_note_text
        })
        
    os.makedirs(out_dir, exist_ok=True)
    vec_path = os.path.join(out_dir, "synthetic_patient_vectors.json")
    notes_path = os.path.join(out_dir, "synthetic_clinical_notes.json")
    
    with open(vec_path, "w") as f:
        json.dump(patients, f, indent=4)
        
    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=4)
        
    print(f"[SUCCESS] Generated {len(patients)} synthetic patient vectors at {vec_path}")
    print(f"[SUCCESS] Generated {len(notes)} synthetic clinical notes at {notes_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic Edge Cases")
    parser.add_argument("--num_cases", type=int, default=20, help="Number of cases to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "../data/synthetic_outputs"))
    args = parser.parse_args()
    
    generate_edge_cases(args.num_cases, args.seed, args.output_dir)
