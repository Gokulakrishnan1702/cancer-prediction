import os
import json
import numpy as np
from scipy.stats import wasserstein_distance, entropy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIST_PATH = os.path.join(BASE_DIR, "data", "process_seeds", "baseline_dists.json")
SYNTH_PATH = os.path.join(BASE_DIR, "data", "synthetic_outputs", "synthetic_patient_vectors_validated.json")
OUT_PATH = os.path.join(BASE_DIR, "reports", "divergence_metrics.json")


def run_distribution_audit():
    print("[EDA DIVERGENCE] Starting Distribution Divergence Audit (Wasserstein & KL)...")
    if not os.path.exists(BASE_DIST_PATH) or not os.path.exists(SYNTH_PATH):
        print("[EDA DIVERGENCE ERROR] Required input baseline or synthetic outputs missing.")
        return {}

    with open(BASE_DIST_PATH, "r") as f:
        baselines = json.load(f)
    with open(SYNTH_PATH, "r") as f:
        patients = json.load(f)


    synth_labs = {"alt_u_l": [], "ast_u_l": [], "total_bilirubin_mg_dl": []}
    synth_vafs = []

    for p in patients:
        for k in synth_labs.keys():
            v = p["organ_impairment_baseline"].get(k)
            if v is not None:
                synth_labs[k].append(v)
        if p["trajectory_t0_to_t4"]:
            synth_vafs.append(p["trajectory_t0_to_t4"][0].get("ctdna_vaf", 0))

    divergence_metrics = {"biomarkers": {}, "mode_collapse": {}}

    for k in synth_labs.keys():
        if not synth_labs[k]:
            continue
        b_mean = baselines.get("labs", {}).get("means", {}).get(k, 40)
        cov_val = baselines.get("labs", {}).get("covariance", {})
        b_std = np.sqrt(cov_val.get(k, {}).get(k, 100)) if isinstance(cov_val, dict) else 10

        np.random.seed(42)
        base_sample = np.random.normal(b_mean, b_std, len(synth_labs[k]))
        w_dist = float(wasserstein_distance(base_sample, synth_labs[k]))

        # Proxy KL Divergence via histograms
        hist_base, bin_edges = np.histogram(base_sample, bins=10, density=True)
        hist_synth, _ = np.histogram(synth_labs[k], bins=bin_edges, density=True)
        hist_base += 1e-6
        hist_synth += 1e-6
        kl_div = float(entropy(hist_synth, hist_base))

        divergence_metrics["biomarkers"][k] = {
            "wasserstein_dist": w_dist,
            "kl_divergence": kl_div,
            "synth_mean": float(np.mean(synth_labs[k])),
            "synth_var": float(np.var(synth_labs[k]))
        }

    if synth_vafs:
        base_vaf = np.random.uniform(0.01, 0.15, len(synth_vafs))
        w_vaf = float(wasserstein_distance(base_vaf, synth_vafs))
        divergence_metrics["biomarkers"]["ctdna_vaf"] = {"wasserstein_dist": w_vaf}

    # Mode Collapse Audit
    unique_profiles = len(set(json.dumps(p["organ_impairment_baseline"], sort_keys=True) for p in patients))
    divergence_metrics["mode_collapse"] = {
        "uniqueness_ratio": float(unique_profiles / max(len(patients), 1)),
        "total_samples": len(patients),
        "unique_samples": unique_profiles
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(divergence_metrics, f, indent=4)

    print(f"[EDA DIVERGENCE SUCCESS] Audited divergence for {len(synth_labs)} biomarkers -> {OUT_PATH}")
    return divergence_metrics


if __name__ == "__main__":
    run_distribution_audit()
