import os
import pandas as pd
import numpy as np
import json

def generate_dummy_csvs():
    """Generates dummy CSVs for Stages 1-4 if they don't exist, to simulate historical data."""
    print("[INFO] Checking for existing Stage 1-4 output CSVs...")
    
    # Define paths
    paths = {
        "stage1": "../../stage_01_toxicity/data/outputs/toxicity_labels.csv",
        "stage2": "../../stage_02_progression/data/outputs/genomics_vaf.csv",
        "stage3": "../../stage_03_multimodal/data/outputs/labs_longitudinal.csv",
        "stage4": "../../stage_04_slm/data/outputs/slm_decisions.csv",
    }
    
    # Create directories if needed
    for path in paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    # Generate Stage 1
    if not os.path.exists(paths["stage1"]):
        pd.DataFrame({
            "patient_id": range(100),
            "toxicity_grade": np.random.choice([0, 1, 2, 3, 4], 100, p=[0.5, 0.3, 0.1, 0.08, 0.02])
        }).to_csv(paths["stage1"], index=False)
        
    # Generate Stage 2
    if not os.path.exists(paths["stage2"]):
        mutations = ["EGFR", "KRAS", "TP53", "MET", "ALK"]
        data2 = {"patient_id": range(100), "ctdna_vaf": np.random.uniform(0.1, 15.0, 100)}
        for mut in mutations:
            data2[mut] = np.random.choice([0, 1], 100, p=[0.8, 0.2])
        pd.DataFrame(data2).to_csv(paths["stage2"], index=False)
        
    # Generate Stage 3
    if not os.path.exists(paths["stage3"]):
        pd.DataFrame({
            "patient_id": range(100),
            "alt_u_l": np.random.normal(40, 15, 100),
            "ast_u_l": np.random.normal(35, 10, 100),
            "total_bilirubin_mg_dl": np.random.normal(1.0, 0.4, 100)
        }).to_csv(paths["stage3"], index=False)
        
    # Generate Stage 4
    if not os.path.exists(paths["stage4"]):
        pd.DataFrame({
            "patient_id": range(100),
            "slm_decision": ["Continue Treatment"] * 100
        }).to_csv(paths["stage4"], index=False)

def extract_baselines():
    print("[INFO] Starting extraction of baseline distributions...")
    
    stage1_path = "../../stage_01_toxicity/data/outputs/toxicity_labels.csv"
    stage2_path = "../../stage_02_progression/data/outputs/genomics_vaf.csv"
    stage3_path = "../../stage_03_multimodal/data/outputs/labs_longitudinal.csv"
    
    try:
        df1 = pd.read_csv(stage1_path)
        df2 = pd.read_csv(stage2_path)
        df3 = pd.read_csv(stage3_path)
        print("[SUCCESS] Loaded all upstream CSV artifacts.")
    except Exception as e:
        print(f"[ERROR] Failed to load CSVs: {e}")
        return

    # Extract Lab distributions
    labs = ["alt_u_l", "ast_u_l", "total_bilirubin_mg_dl"]
    lab_means = df3[labs].mean().to_dict()
    lab_cov = df3[labs].cov().to_dict()
    
    ctdna_mean = df2["ctdna_vaf"].mean()
    ctdna_var = df2["ctdna_vaf"].var()
    
    # Extract Mutation Frequencies
    mutations = ["EGFR", "KRAS", "TP53", "MET", "ALK"]
    mut_freqs = df2[mutations].mean().to_dict()
    
    # Extract Toxicity Grade Probabilities
    tox_counts = df1["toxicity_grade"].value_counts(normalize=True).to_dict()
    # Ensure all grades 0-4 are present
    tox_probs = {str(k): v for k, v in tox_counts.items()}
    
    baseline_dists = {
        "labs": {
            "means": lab_means,
            "covariance": lab_cov
        },
        "ctdna": {
            "mean": ctdna_mean,
            "variance": ctdna_var
        },
        "mutations": {
            "frequencies": mut_freqs
        },
        "toxicity": {
            "grade_probabilities": tox_probs
        }
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "../data/process_seeds/baseline_dists.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(baseline_dists, f, indent=4)
        
    print(f"[SUCCESS] Exported baseline distributions to {output_path}")

if __name__ == "__main__":
    generate_dummy_csvs()
    extract_baselines()
