import os
import json
import random
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SEEDS_PATH = os.path.join(DATA_DIR, "process_seeds", "baseline_dists.json")
OUTPUTS_DIR = os.path.join(DATA_DIR, "synthetic_outputs")


def generate_20_edge_cases(num_cases: int = 20, seed: int = 42, out_dir: Optional[str] = None):
    print(f"[GENAI EDGE] Generating {num_cases} Out-of-Distribution Edge Scenarios (Seed: {seed})...")
    random.seed(seed)

    if out_dir is None:
        out_dir = OUTPUTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    patients = []
    notes = []

    dili_count = int(num_cases * 0.3)
    dili_indices = set(random.sample(range(num_cases - 1), dili_count)) if num_cases > 1 else set()
    dili_indices.add(num_cases - 1)  # SYNTH_EDGE_020 is always severe DILI capstone trap

    for i in range(num_cases):
        patient_id = f"SYNTH_EDGE_{i+1:03d}"

        # Genomics
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
            alt = round(random.uniform(15.0, 45.0), 1)
            ast = round(random.uniform(15.0, 40.0), 1)
            bilirubin = round(random.uniform(0.4, 1.2), 1)
            tox_grade = 1

        # Trajectories
        trajectory = []
        ctdna_base = random.uniform(0.005, 0.05)
        tumor_base = random.uniform(0.02, 0.08)

        is_progression = ("MET amplification" in mutations) or ("C797S" in " ".join(mutations))
        multiplier = random.uniform(1.15, 1.30) if is_progression else random.uniform(0.75, 0.88)

        for t in range(5):
            trajectory.append({
                "timepoint": f"T{t}",
                "ctdna_vaf": round(min(1.0, ctdna_base * (multiplier ** t)), 4),
                "tumor_vol_cm3": round(tumor_base * (multiplier ** t), 3)
            })

        patient_record = {
            "synthetic_patient_id": patient_id,
            "is_ood": True,
            "baseline_genomics": mutations,
            "organ_impairment_baseline": {
                "alt_u_l": alt,
                "ast_u_l": ast,
                "total_bilirubin_mg_dl": bilirubin,
                "creatinine_mg_dl": round(random.uniform(0.8, 1.3), 2)
            },
            "toxicity_grade": tox_grade,
            "is_severe_dili": is_dili,
            "trajectory_t0_to_t4": trajectory,
            "metadata": {"phenotype": "OOD Edge Scenario"}
        }
        patients.append(patient_record)

        # Generate clinical progress note
        note_text = (
            f"Clinical encounter for {patient_id}. Identified mutations: {', '.join(mutations)}. "
            f"Baseline organ function assessment: AST {ast} U/L, ALT {alt} U/L, Tbili {bilirubin} mg/dL. "
            f"Initial tumor burden: {trajectory[0]['tumor_vol_cm3']} cm3 with ctDNA fraction {trajectory[0]['ctdna_vaf']}%. "
            f"By timepoint T4, tumor burden is {trajectory[-1]['tumor_vol_cm3']} cm3. "
            f"{'Patient experienced Grade 3 severe Drug-Induced Liver Injury (DILI).' if is_dili else 'No severe hepatic toxicity noted.'}"
        )
        notes.append({
            "synthetic_patient_id": patient_id,
            "clinical_note": note_text
        })

    with open(os.path.join(out_dir, "synthetic_patient_vectors.json"), "w") as f:
        json.dump(patients, f, indent=4)
    with open(os.path.join(out_dir, "synthetic_clinical_notes.json"), "w") as f:
        json.dump(notes, f, indent=4)

    print(f"[GENAI EDGE SUCCESS] Generated and saved {len(patients)} edge cases to {out_dir}")
    return patients, notes


if __name__ == "__main__":
    generate_20_edge_cases()
