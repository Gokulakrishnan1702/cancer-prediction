import os
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "synthetic_outputs", "synthetic_patient_vectors_validated.json")
OUT_PATH = os.path.join(BASE_DIR, "reports", "latent_metrics.json")


def run_latent_profiling():
    print("[EDA LATENT] Starting Latent Space Profiling & Outlier Detection...")
    if not os.path.exists(INPUT_PATH):
        print(f"[EDA ERROR] Input file not found: {INPUT_PATH}")
        return {}

    with open(INPUT_PATH, "r") as f:
        patients = json.load(f)

    features = []
    compound_mutations = 0

    for p in patients:
        labs = p["organ_impairment_baseline"]
        traj = p["trajectory_t0_to_t4"]

        alt = labs.get("alt_u_l") or 0
        ast = labs.get("ast_u_l") or 0
        bili = labs.get("total_bilirubin_mg_dl") or 0
        vaf = traj[0].get("ctdna_vaf", 0) if traj else 0
        vol = traj[0].get("tumor_vol_cm3", 0) if traj else 0

        features.append([alt, ast, bili, vaf, vol])

        gens = " ".join(p["baseline_genomics"])
        if ("EGFR" in gens and "MET" in gens) or ("KRAS" in gens and "MET" in gens) or ("EGFR" in gens and "C797S" in gens and "T790M" in gens):
            compound_mutations += 1

    X = np.array(features)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    centroid = np.mean(X_pca, axis=0)
    distances = np.linalg.norm(X_pca - centroid, axis=1)
    std_dist = np.std(distances)

    ood_count = int(np.sum(distances > (np.mean(distances) + 2.5 * std_dist)))

    metrics = {
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
        "ood_samples_count": ood_count,
        "compound_mutation_cases": compound_mutations,
        "total_cases": len(patients)
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[EDA LATENT SUCCESS] Completed profiling: {ood_count} OOD cases, {compound_mutations} compound variants -> {OUT_PATH}")
    return metrics


if __name__ == "__main__":
    run_latent_profiling()
