import os
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SEEDS_DIR = os.path.join(DATA_DIR, "process_seeds")
RAW_SEEDS_DIR = os.path.join(DATA_DIR, "raw_seeds")


def generate_dummy_csvs():
    """Generates baseline distribution seeds if upstream stages are not yet fully populated."""
    print("[DATA ENG] Checking for upstream baseline distribution seeds...")
    os.makedirs(RAW_SEEDS_DIR, exist_ok=True)
    os.makedirs(SEEDS_DIR, exist_ok=True)

    stage1_path = os.path.join(RAW_SEEDS_DIR, "toxicity_labels.csv")
    stage2_path = os.path.join(RAW_SEEDS_DIR, "genomics_vaf.csv")
    stage3_path = os.path.join(RAW_SEEDS_DIR, "labs_longitudinal.csv")

    if not os.path.exists(stage1_path):
        pd.DataFrame({
            "patient_id": range(100),
            "toxicity_grade": np.random.choice([0, 1, 2, 3, 4], 100, p=[0.5, 0.3, 0.1, 0.08, 0.02])
        }).to_csv(stage1_path, index=False)

    if not os.path.exists(stage2_path):
        mutations = ["EGFR", "KRAS", "TP53", "MET", "ALK"]
        data2 = {"patient_id": range(100), "ctdna_vaf": np.random.uniform(0.1, 15.0, 100)}
        for mut in mutations:
            data2[mut] = np.random.choice([0, 1], 100, p=[0.8, 0.2])
        pd.DataFrame(data2).to_csv(stage2_path, index=False)

    if not os.path.exists(stage3_path):
        pd.DataFrame({
            "patient_id": range(100),
            "alt_u_l": np.random.normal(40, 15, 100),
            "ast_u_l": np.random.normal(35, 10, 100),
            "total_bilirubin_mg_dl": np.random.normal(1.0, 0.4, 100)
        }).to_csv(stage3_path, index=False)


def extract_baselines():
    """Extracts empirical baseline parameter distributions for VAE and LLM prompting."""
    print("[DATA ENG] Starting extraction of baseline distributions...")
    generate_dummy_csvs()

    stage1_path = os.path.join(RAW_SEEDS_DIR, "toxicity_labels.csv")
    stage2_path = os.path.join(RAW_SEEDS_DIR, "genomics_vaf.csv")
    stage3_path = os.path.join(RAW_SEEDS_DIR, "labs_longitudinal.csv")

    try:
        df1 = pd.read_csv(stage1_path)
        df2 = pd.read_csv(stage2_path)
        df3 = pd.read_csv(stage3_path)
        print("[DATA ENG] Loaded upstream seeds successfully.")
    except Exception as e:
        print(f"[DATA ENG ERROR] Failed to load seeds: {e}")
        return

    labs = ["alt_u_l", "ast_u_l", "total_bilirubin_mg_dl"]
    lab_means = df3[labs].mean().to_dict()
    lab_cov = df3[labs].cov().to_dict()

    ctdna_mean = float(df2["ctdna_vaf"].mean())
    ctdna_var = float(df2["ctdna_vaf"].var())

    mutations = ["EGFR", "KRAS", "TP53", "MET", "ALK"]
    mut_freqs = df2[mutations].mean().to_dict()

    tox_counts = df1["toxicity_grade"].value_counts(normalize=True).to_dict()
    tox_probs = {str(k): float(v) for k, v in tox_counts.items()}

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

    output_path = os.path.join(SEEDS_DIR, "baseline_dists.json")
    with open(output_path, "w") as f:
        json.dump(baseline_dists, f, indent=4)

    print(f"[DATA ENG SUCCESS] Exported baseline distributions to {output_path}")
    return baseline_dists


if __name__ == "__main__":
    extract_baselines()
